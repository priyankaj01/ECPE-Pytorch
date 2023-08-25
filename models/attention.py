import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(True),
            nn.Linear(2 * hidden_dim, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


def main():
    a = torch.rand(32, 12, 128)
    print(a.shape)
    att = Attention(128)
    a, weight = att(a)
    print(a.shape)
    print(weight)


if __name__ == '__main__':
    main()
