# model_itransformer.py
import torch
import torch.nn as nn

class iTransformerClassifier(nn.Module):
    """
    输入: (B, L=100, F)
    1) invert -> (B, F, L)
    2) per-variable token embedding: Linear(L -> d_model)
    3) TransformerEncoder over tokens (sequence length = F)
    4) mean pool over tokens -> classifier (3 classes)
    """
    def __init__(self, n_vars: int, seq_len: int = 100, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_embed = nn.Linear(seq_len, d_model)

        # 可选的“变量位置/身份嵌入”
        self.var_embed = nn.Parameter(torch.zeros(1, n_vars, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, 3)

    def forward(self, x):
        # x: (B, L, F)
        x = x.transpose(1, 2)          # (B, F, L)
        tok = self.token_embed(x)      # (B, F, d_model)
        tok = tok + self.var_embed
        tok = self.encoder(tok)        # (B, F, d_model)
        tok = self.norm(tok)
        pooled = tok.mean(dim=1)       # (B, d_model)
        logits = self.cls(pooled)      # (B, 3)
        return logits
