import torch
import torch.nn as nn

__all__ = ["create_tabular_transformer_model"]


def _set_dropouts_on_encoder_layer(layer: nn.TransformerEncoderLayer,
                                   attn_p: float,
                                   ffn_p: float,
                                   residual_p: float):
    """
    PyTorch 버전 차이를 고려해 안전하게 드롭아웃을 설정.
    - self_attn 드롭아웃
    - FFN 드롭아웃 (dropout1)
    - Residual 경로 드롭아웃 (dropout2 또는 dropout)
    """
    # 어텐션 드롭아웃
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "dropout"):
        layer.self_attn.dropout = nn.Dropout(p=attn_p)

    # PyTorch 2.x: dropout1, dropout2 존재
    if hasattr(layer, "dropout1"):
        layer.dropout1.p = ffn_p
    if hasattr(layer, "dropout2"):
        layer.dropout2.p = residual_p

    # 예전 버전 호환: 단일 dropout 필드가 있을 수 있음
    if hasattr(layer, "dropout") and not hasattr(layer, "dropout2"):
        layer.dropout.p = residual_p


class TabularTransformerModel(nn.Module):
    def __init__(
        self,
        num_categorical_features=0,
        categorical_cardinalities=None,  # 각 피처마다 +1(OOV/미싱) 까지 포함된 값이어야 함
        num_numerical_features=0,
        lstm_hidden=32,
        hidden_dim=192,
        n_heads=8,
        n_layers=3,
        ffn_size_factor=4/3,
        attention_dropout=0.2,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        device="cuda",
    ):
        super().__init__()

        self.num_categorical_features = num_categorical_features
        self.num_numerical_features = num_numerical_features
        self.hidden_dim = hidden_dim
        self.device = device

        # Categorical embeddings (+1 인덱스 체계 전제; 0은 OOV/미싱 예약)
        if num_categorical_features > 0 and categorical_cardinalities is not None:
            assert len(categorical_cardinalities) == num_categorical_features, \
                "categorical_cardinalities 길이가 num_categorical_features와 일치해야 합니다."
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, hidden_dim)  # cardinality는 +1이 반영된 값이어야 함
                for cardinality in categorical_cardinalities
            ])
        else:
            self.categorical_embeddings = None

        # Numerical feature projection (각 수치 1→hidden)
        self.numerical_projection = nn.Linear(1, hidden_dim) if num_numerical_features > 0 else None

        # Sequential features (LSTM)
        if lstm_hidden > 0:
            self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
            self.seq_projection = nn.Linear(lstm_hidden, hidden_dim)
        else:
            self.lstm = None
            self.seq_projection = None

        # 총 feature 토큰 수 계산 (cat + num + seq(있으면 1))
        total_features = 0
        if num_categorical_features > 0:
            total_features += num_categorical_features
        if num_numerical_features > 0:
            total_features += num_numerical_features
        if lstm_hidden > 0:
            total_features += 1  # LSTM 출력 1개 피처로 간주

        # Column(포지션) 임베딩/클래스 토큰/NaN 토큰
        self.column_embeddings = nn.Parameter(torch.zeros(total_features, hidden_dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.nan_token = nn.Parameter(torch.zeros(1, hidden_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=int(hidden_dim * ffn_size_factor),
            dropout=residual_dropout,   # residual 경로 기본 드롭아웃
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        _set_dropouts_on_encoder_layer(
            encoder_layer,
            attn_p=attention_dropout,
            ffn_p=ffn_dropout,
            residual_p=residual_dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """트렁크 정규분포 초기화 (파라미터 포함)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LSTM):
                # LSTM 가중치도 가볍게 초기화
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.trunc_normal_(param, std=0.02)
                    elif "bias" in name:
                        nn.init.zeros_(param)

        # 개별 파라미터(모듈이 아님)
        nn.init.trunc_normal_(self.column_embeddings, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.nan_token, std=0.02)

    def forward(self, x_categorical=None, x_numerical=None, x_seq=None, seq_lengths=None, nan_mask=None):
        batch_size = None
        features = []
        feature_idx = 0

        # Categorical features
        if self.categorical_embeddings is not None and x_categorical is not None and x_categorical.numel() > 0:
            batch_size = x_categorical.size(0)
            for i, embedding in enumerate(self.categorical_embeddings):
                cat_idx = x_categorical[:, i]  # (B,)  0==OOV/미싱 예약
                cat_feature = embedding(cat_idx)  # (B, hidden_dim)

                # 결측/OOV 마스크: (nan_mask OR idx==0)
                if nan_mask is not None:
                    miss_from_mask = nan_mask[:, feature_idx].bool()  # (B,)
                    miss_from_oov = (cat_idx == 0)
                    missing_mask = (miss_from_mask | miss_from_oov)
                else:
                    missing_mask = (cat_idx == 0)

                if missing_mask.any():
                    cat_feature = torch.where(
                        missing_mask.unsqueeze(-1),
                        self.nan_token.expand(cat_feature.size(0), -1),
                        cat_feature
                    )

                features.append(cat_feature)
                feature_idx += 1

        # Numerical features
        if self.numerical_projection is not None and x_numerical is not None and x_numerical.numel() > 0:
            if batch_size is None:
                batch_size = x_numerical.size(0)
            # (B, num_numerical_features, hidden_dim)
            num_features = self.numerical_projection(x_numerical.unsqueeze(-1)).squeeze(-2)

            # 결측치 NaN 토큰 대체
            if nan_mask is not None:
                miss = nan_mask[:, feature_idx:feature_idx + self.num_numerical_features].bool()  # (B, N)
                miss = miss.unsqueeze(-1)  # (B, N, 1)
                nan_tokens = self.nan_token.expand(batch_size, self.num_numerical_features, -1)
                num_features = torch.where(miss, nan_tokens, num_features)

            for i in range(self.num_numerical_features):
                features.append(num_features[:, i, :])

            feature_idx += self.num_numerical_features

        # Sequential feature
        if self.lstm is not None and x_seq is not None and seq_lengths is not None:
            if batch_size is None:
                batch_size = x_seq.size(0)
            x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            h = h_n[-1]  # (B, lstm_hidden)
            seq_feature = self.seq_projection(h)  # (B, hidden_dim)

            # 결측 마스크 적용
            if nan_mask is not None:
                missing_mask = nan_mask[:, feature_idx].bool()
                if missing_mask.any():
                    seq_feature = torch.where(
                        missing_mask.unsqueeze(-1),
                        self.nan_token.expand(batch_size, -1),
                        seq_feature
                    )

            features.append(seq_feature)
            feature_idx += 1

        if not features:
            raise ValueError("No features provided to the model")

        # Stack & add column embeddings
        x = torch.stack(features, dim=1)  # (B, F, H)
        x = x + self.column_embeddings[:len(features)].unsqueeze(0)  # (1, F, H)

        # Class token prepend
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # (B, 1, H)
        x = torch.cat([class_tokens, x], dim=1)  # (B, 1+F, H)

        # Transformer
        x = self.transformer(x)  # (B, 1+F, H)

        # CLS
        class_output = x[:, 0, :]  # (B, H)
        output = self.output_layer(class_output)  # (B, 1)

        return output.squeeze(1)  # (B,)


def create_tabular_transformer_model(
    num_categorical_features=0,
    categorical_cardinalities=None,
    num_numerical_features=0,
    lstm_hidden=32,
    hidden_dim=192,
    n_heads=8,
    n_layers=3,
    ffn_size_factor=4/3,
    attention_dropout=0.2,
    ffn_dropout=0.1,
    residual_dropout=0.0,
    device="cuda",
):
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
        device=device,
    ).to(device)
    return model
