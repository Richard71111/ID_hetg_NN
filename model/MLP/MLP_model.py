import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A fully connected residual block:
    y = proj(x) + Linear -> (LayerNorm) -> activation -> (Dropout)

    - If in_dim == out_dim: residual connection is added directly.
    - If in_dim != out_dim: a Linear projection aligns the dimensions.
    """
    def __init__(self, in_dim, out_dim, activation=nn.SiLU, dropout=0.1, layernorm=True):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim) if layernorm else nn.Identity(),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )


        # Add a projection if input and output dimensions differ
        self.proj = None
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        out = self.net(x)
        return residual + out


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with optional residual connections on each layer.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dims (list[int]): List of hidden layer sizes.
        output_dim (int): Dimension of the output layer.
        activation (nn.Module): Activation function class (default: nn.SiLU).
        dropout (float): Dropout rate (default: 0.1).
        layernorm (bool): Whether to apply LayerNorm after each Linear layer.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 activation=nn.SiLU, dropout=0.1, layernorm=True):
        super().__init__()

        blocks = []
        prev_dim = input_dim

        # Each hidden dimension corresponds to one ResidualBlock
        for h_dim in hidden_dims:
            blocks.append(
                ResidualBlock(
                    in_dim=prev_dim,
                    out_dim=h_dim,
                    activation=activation,
                    dropout=dropout,
                    layernorm=layernorm
                )
            )
            prev_dim = h_dim

        # Stack all blocks sequentially
        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.out(x)
        return x
