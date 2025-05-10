# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# your utils.ALL_NUM will be imported by train script
# EMBEDS = ["track_name","car_name"]
# ALL_NUM  = list of all  numeric sensor names

class TORCSModel(nn.Module):
    def __init__(self,
                 num_features: int,
                 emb_sizes: dict[str,int],
                 seq_len: int = 5,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 2):
        """
        num_features: len(ALL_NUM)
        emb_sizes:    {"track_name": n_tracks, "car_name": n_cars}
        seq_len:      how many past frames to use
        """
        super().__init__()
        self.seq_len = seq_len

        # richer embeddings
        self.emb_track = nn.Embedding(emb_sizes["track_name"], 16)
        self.emb_car   = nn.Embedding(emb_sizes["car_name"],   8)
        emb_dim = 16 + 8

        # LSTM to capture temporal patterns
        self.lstm = nn.LSTM(input_size=num_features + emb_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=0.3)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)        # steer, accel, brake
        )

    def forward(self,
                x_num: torch.Tensor,
                x_cat: torch.LongTensor):
        """
        x_num: (B, seq_len, num_features) or (B, num_features)
        x_cat: (B, 2)            — track & car indices
        """
        B = x_cat.size(0)

        # build embeddings and tile across seq_len
        emb_t = self.emb_track(x_cat[:,0])    # (B,16)
        emb_c = self.emb_car(  x_cat[:,1])    # (B, 8)
        emb   = torch.cat([emb_t, emb_c], 1)  # (B,24)

        # if single‑frame input, promote to (B,1,F)
        if x_num.dim() == 2:
            x = x_num.unsqueeze(1)            # (B,1,F)
            sl = 1
        else:
            x = x_num                         # (B,seq_len,F)
            sl = x.size(1)

        # append embeddings to every time step
        emb_seq = emb.unsqueeze(1).expand(-1, sl, -1)          # (B,sl,24)
        lstm_in = torch.cat([x, emb_seq], dim=2)               # (B,sl, F+24)

        # LSTM
        out, (h_n, _) = self.lstm(lstm_in)                     # out: (B,sl,H), h_n: (L,B,H)
        feat = h_n[-1]                                         # take last layer’s final hidden: (B,H)

        # MLP head
        raw = self.head(feat)                                  # (B,3)
        steer = torch.tanh(raw[:,0])                          # [-1,1]
        accel = torch.sigmoid(raw[:,1])                       # [0,1]
        brake = torch.sigmoid(raw[:,2])                       # [0,1]

        return torch.stack([steer, accel, brake], dim=1)      # (B,3)
