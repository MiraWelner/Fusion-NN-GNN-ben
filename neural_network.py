import torch
import torch.nn as nn


# Sub-network A
class heartrate_net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        return self.net(x)

# Sub-network B
class ecg_net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        return self.net(x)

class combo_net(nn.Module):
    def __init__(self, output_len):
        super().__init__()
        self.output_len = output_len
        self.hr_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU()
        )
        self.ecg_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, output_len*2)
        )

    def forward(self, a_out, b_out):
        a = self.hr_proj(a_out)
        b = self.ecg_proj(b_out)
        fused = torch.cat([a, b], dim=1)
        output = self.fusion(fused)
        ecg_output = output[:,self.output_len:]
        rr_output = output[:,:self.output_len]
        return ecg_output, rr_output

class GNN(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.a = heartrate_net(input_dim=signal_length)
        self.b = ecg_net(input_dim=signal_length)
        self.fusion = combo_net(output_len=signal_length)

    def forward(self, x1, x2):
        out_a = self.a(x1)
        out_b = self.b(x2)
        return self.fusion(out_a, out_b)
