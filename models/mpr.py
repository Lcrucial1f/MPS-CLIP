import torch.nn as nn


def l2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class MultiPerspectiveRepresentation(nn.Module):
    """Project an aggregated subimage feature into complementary perspectives."""

    def __init__(self, in_dim, out_dim=None, k=4, mlp=True, dropout=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        self.k = k
        self.heads = nn.ModuleList()
        self.dropout_p = dropout

        for _ in range(k):
            if mlp:
                layers = [
                    nn.Linear(in_dim, in_dim),
                    nn.GELU(),
                ]
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(in_dim, out_dim))
                self.heads.append(nn.Sequential(*layers))
            else:
                if dropout > 0.0:
                    self.heads.append(
                        nn.Sequential(
                            nn.Linear(in_dim, out_dim),
                            nn.Dropout(p=dropout),
                        )
                    )
                else:
                    self.heads.append(nn.Linear(in_dim, out_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for head in self.heads:
            v = head(x)
            v = l2norm(v, dim=-1)
            outs.append(v)
        return outs
