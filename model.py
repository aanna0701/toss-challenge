import torch
import torch.nn as nn

__all__ = ["create_tabular_transformer_model"]



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
        
        # NaN token for missing values
        self.nan_token = nn.Parameter(torch.zeros(1, hidden_dim))
        
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
                
    def forward(self, x_categorical=None, x_numerical=None, x_seq=None, seq_lengths=None, nan_mask=None):
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
        
        # Add column embeddings BEFORE missing value masking
        x = x + self.column_embeddings.unsqueeze(0)

        # Handle missing values AFTER column embeddings
        if nan_mask is not None:
            # Create NaN token without column embeddings (just the base NaN token)
            nan_tokens = self.nan_token.expand(batch_size, len(features), -1)
            
            # Apply missing value masking
            missing_masks = nan_mask.bool().unsqueeze(-1)  # (B, num_features, 1)
            x = torch.where(missing_masks, nan_tokens, x)
        
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