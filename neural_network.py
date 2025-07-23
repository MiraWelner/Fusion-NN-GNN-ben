import torch
import torch.nn as nn


# Sub-network A
class heartrate_net(nn.Module):
    def __init__(self):
        super(heartrate_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3000, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.net(x)

# Sub-network B
class ecg_net(nn.Module):
    def __init__(self):
        super(ecg_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3000, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.net(x)

# Fusion Network
class combo_net(nn.Module):
    def __init__(self, output_len=3000):
        super(combo_net, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.out1 = nn.Linear(16, output_len)
        self.out2 = nn.Linear(16, output_len)

    def forward(self, a_out, b_out):
        combined = torch.cat((a_out, b_out), dim=1)
        fused = self.shared(combined)
        return self.out1(fused), self.out2(fused)

# Combine everything into a model wrapper
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.a = heartrate_net()
        self.b = ecg_net()
        self.fusion = combo_net()

    def forward(self, x1, x2):
        out_a = self.a(x1)
        out_b = self.b(x2)
        return self.fusion(out_a, out_b)  # returns y1_pred, y2_pred
