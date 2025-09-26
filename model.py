import torch
import torch.nn as nn

__all__ = [
    "create_tabular_seq_model"
    ]


class TabularSeqModel(nn.Module):
    def __init__(self, d_features, lstm_hidden=32, hidden_units=[1024, 512, 256, 128], dropout=0.2):
        super().__init__()
        # 모든 비-시퀀스 피처에 BN
        self.bn_x = nn.BatchNorm1d(d_features)
        # seq: 숫자 시퀀스 → LSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)

        # 최종 MLP
        input_dim = d_features + lstm_hidden
        layers = []
        for h in hidden_units:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq, seq_lengths):
        # 비-시퀀스 피처
        x = self.bn_x(x_feats)

        # 시퀀스 → LSTM (pack)
        x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]                  # (B, lstm_hidden)

        z = torch.cat([x, h], dim=1)
        return self.mlp(z).squeeze(1)  # logits

def create_tabular_seq_model(d_features, lstm_hidden=64, hidden_units=[256, 128], dropout=0.2, device="cuda"):
    """모델 생성 함수"""
    model = TabularSeqModel(
        d_features=d_features, 
        lstm_hidden=lstm_hidden, 
        hidden_units=hidden_units, 
        dropout=dropout
    ).to(device)
    return model
