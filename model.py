import torch
import torch.nn as nn

__all__ = ["create_tabular_transformer_model", "create_widedeep_ctr_model"]


class CrossNetwork(nn.Module):
    """Cross Network for WideDeepCTR model"""
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        x0 = x  # Save the original input
        for layer in self.cross_layers:
            x = x0 * layer(x) + x  # Cross layer formula: x0 * (W * x + b) + x
        return x



class TabularTransformerModel(nn.Module):
    def __init__(self, 
                 num_categorical_features, 
                 categorical_cardinalities,
                 num_numerical_features,
                 lstm_hidden,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 ffn_size_factor,
                 attention_dropout,
                 ffn_dropout,
                 residual_dropout,
                 device):
        super().__init__()
        
        self.num_categorical_features = num_categorical_features
        self.num_numerical_features = num_numerical_features
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Categorical embeddings
        if num_categorical_features > 0 and categorical_cardinalities is not None:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, hidden_dim) 
                for cardinality in categorical_cardinalities
            ])
        else:
            self.categorical_embeddings = None
            
        # Numerical feature projection
        if num_numerical_features > 0:
            self.numerical_projection = nn.Linear(1, hidden_dim)
        else:
            self.numerical_projection = None
            
        # Sequential features (LSTM)
        if lstm_hidden > 0:
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        else:
            self.lstm = None
            
        # Calculate total number of features for column embeddings
        total_features = 0
        if num_categorical_features > 0:
            total_features += num_categorical_features
        if num_numerical_features > 0:
            total_features += num_numerical_features
        if lstm_hidden > 0:
            total_features += 1  # LSTM output counts as one feature
            
        # Column embeddings
        self.column_embeddings = nn.Parameter(torch.zeros(total_features, hidden_dim))
        
        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=int(hidden_dim * ffn_size_factor),
            dropout=residual_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # Custom attention dropout
        for layer in encoder_layer.self_attn.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = attention_dropout
                
        # Custom FFN dropout
        for layer in encoder_layer.linear1.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = ffn_dropout
        for layer in encoder_layer.linear2.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = ffn_dropout
                
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with truncated normal distribution"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Parameter):
                nn.init.trunc_normal_(module, std=0.02)
                
    def forward(self, x_categorical=None, x_numerical=None, x_seq=None, seq_lengths=None):
        batch_size = None
        features = []
        feature_idx = 0
        
        # Process categorical features
        if self.categorical_embeddings is not None and x_categorical is not None:
            batch_size = x_categorical.size(0)
            for i, embedding in enumerate(self.categorical_embeddings):
                cat_feature = embedding(x_categorical[:, i])  # (B, hidden_dim)
                features.append(cat_feature)
                feature_idx += 1
                
        # Process numerical features
        if self.numerical_projection is not None and x_numerical is not None:
            if batch_size is None:
                batch_size = x_numerical.size(0)
            
            # Project all numerical features at once: (B, num_numerical_features, hidden_dim)
            num_features = self.numerical_projection(x_numerical.unsqueeze(-1)).squeeze(-2)  # (B, num_numerical_features, hidden_dim)
            
            # Add each numerical feature to the features list
            for i in range(self.num_numerical_features):
                features.append(num_features[:, i, :])  # (B, hidden_dim)
            
            feature_idx += self.num_numerical_features
            
        # Process sequential features
        if self.lstm is not None and x_seq is not None and seq_lengths is not None:
            if batch_size is None:
                batch_size = x_seq.size(0)
            # Process sequence through LSTM
            x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            seq_feature = h_n[-1]  # (B, hidden_dim)
            
            features.append(seq_feature)
            feature_idx += 1
            
        if not features:
            raise ValueError("No features provided to the model")
            
        # Stack features: (B, num_features, hidden_dim)
        x = torch.stack(features, dim=1)
        
        # Add column embeddings
        x = x + self.column_embeddings.unsqueeze(0)
        
        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([class_tokens, x], dim=1)  # (B, 1+num_features, hidden_dim)
        
        # Apply transformer
        x = self.transformer(x)  # (B, 1+num_features, hidden_dim)
        
        # Use only class token for prediction
        class_output = x[:, 0, :]  # (B, hidden_dim)
        
        # Final prediction
        output = self.output_layer(class_output)  # (B, 1)
        
        return output.squeeze(1)  # (B,)


class WideDeepCTR(nn.Module):
    """WideDeepCTR model for click-through rate prediction"""
    def __init__(self, num_features, cat_cardinalities, emb_dim=16, lstm_hidden=64,
                 hidden_units=[512, 256, 128], dropout=[0.1, 0.2, 0.3], device='cpu'):
        super().__init__()
        self.device = device
        
        # Categorical embeddings
        if len(cat_cardinalities) > 0:
            self.emb_layers = nn.ModuleList([
                nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities
            ])
        else:
            self.emb_layers = None
            
        # Calculate dimensions
        cat_input_dim = emb_dim * len(cat_cardinalities) if cat_cardinalities else 0
        
        # Batch normalization for numerical features
        if num_features > 0:
            self.bn_num = nn.BatchNorm1d(num_features)
        else:
            self.bn_num = None
            
        # LSTM for sequential features
        if lstm_hidden > 0:
            self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden,
                                num_layers=2, batch_first=True, bidirectional=True)
            seq_out_dim = lstm_hidden * 2
        else:
            self.lstm = None
            seq_out_dim = 0
            
        # Cross Network
        cross_input_dim = num_features + cat_input_dim + seq_out_dim
        if cross_input_dim > 0:
            self.cross = CrossNetwork(cross_input_dim, num_layers=2)
        else:
            self.cross = None
            
        # Deep MLP
        input_dim = cross_input_dim
        if input_dim > 0 and hidden_units:
            layers = []
            for i, h in enumerate(hidden_units):
                layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout[i % len(dropout)])]
                input_dim = h
            layers += [nn.Linear(input_dim, 1)]
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None
            
    def forward(self, num_x=None, cat_x=None, seqs=None, seq_lengths=None):
        """
        Forward pass for WideDeepCTR model
        Args:
            num_x: numerical features (B, num_features)
            cat_x: categorical features (B, num_categorical)
            seqs: sequential features (B, seq_length)
            seq_lengths: sequence lengths (B,)
        Returns:
            output: (B,)
        """
        features = []
        
        # Process numerical features
        if self.bn_num is not None and num_x is not None:
            num_x = self.bn_num(num_x)
            features.append(num_x)
            
        # Process categorical features
        if self.emb_layers is not None and cat_x is not None:
            cat_embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)]
            cat_feat = torch.cat(cat_embs, dim=1)
            features.append(cat_feat)
            
        # Process sequential features
        if self.lstm is not None and seqs is not None and seq_lengths is not None:
            seqs = seqs.unsqueeze(-1)  # (B, seq_length, 1)
            packed = nn.utils.rnn.pack_padded_sequence(
                seqs, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            # Concatenate last hidden states from both directions
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
            features.append(h)
            
        if not features:
            raise ValueError("No features provided to the model")
            
        # Concatenate all features
        z = torch.cat(features, dim=1)
        
        # Apply cross network
        if self.cross is not None:
            z_cross = self.cross(z)
        else:
            z_cross = z
            
        # Apply MLP
        if self.mlp is not None:
            out = self.mlp(z_cross)
        else:
            out = z_cross
            
        return out.squeeze(1)  # (B,)


def create_tabular_transformer_model(num_categorical_features, 
                                   categorical_cardinalities,
                                   num_numerical_features,
                                   lstm_hidden,
                                   hidden_dim,
                                   n_heads,
                                   n_layers,
                                   ffn_size_factor,
                                   attention_dropout,
                                   ffn_dropout,
                                   residual_dropout,
                                   device):
    """Tabular Transformer 모델 생성 함수"""
    model = TabularTransformerModel(
        num_categorical_features=num_categorical_features,
        categorical_cardinalities=categorical_cardinalities,
        num_numerical_features=num_numerical_features,
        lstm_hidden=lstm_hidden,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_size_factor=ffn_size_factor,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        residual_dropout=residual_dropout,
        device=device
    ).to(device)
    return model


def create_widedeep_ctr_model(num_features, 
                              cat_cardinalities,
                              emb_dim=16,
                              lstm_hidden=64,
                              hidden_units=[512, 256, 128],
                              dropout=[0.1, 0.2, 0.3],
                              device='cpu'):
    """WideDeepCTR 모델 생성 함수"""
    model = WideDeepCTR(
        num_features=num_features,
        cat_cardinalities=cat_cardinalities,
        emb_dim=emb_dim,
        lstm_hidden=lstm_hidden,
        hidden_units=hidden_units,
        dropout=dropout,
        device=device
    ).to(device)
    return model
