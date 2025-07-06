import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=1024, num_heads=8, num_layers=4, hidden_dim=2048):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: [B, T, 1024]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.regressor(x.mean(dim=1))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)
        
    def forward(self, x):
        return x + self.pe[:x.size(1)]