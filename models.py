import torch.nn as nn
import torch

def build_mlp(num_features, emb_sizes):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_track = nn.Embedding(emb_sizes["track_name"], 8)
            self.emb_car   = nn.Embedding(emb_sizes["car_name"],   4)
            self.net = nn.Sequential(
                nn.Linear(num_features + 12, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 3),   nn.Tanh()
            )
        def forward(self, xn, xc):
            z = torch.cat([xn,
                           self.emb_track(xc[:,0]),
                           self.emb_car(  xc[:,1])], 1)
            out = self.net(z)
            steer = out[:,0]
            accel = torch.clamp(out[:,1], 0, 1)
            brake = torch.clamp(out[:,2], 0, 1)
            return torch.stack([steer, accel, brake], 1)
    return MLP()
