import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size = 3, num_groups = 4) -> None:
        super().__init__()

        padding = (kernel_size - 1) // 2
        # Build convolutional layer
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.gn1   = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act2  = nn.SiLU()
        self.gn2   = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.out_act = nn.SiLU()
    def forward(self, x):
        # x: (B, C, L)
        residual = x
    
        # Block 1: GN -> SiLU -> Conv1D
        out = self.gn1(x)  # Note: Apply GN/SiLU to x first, not the result of conv1
        out = self.act1(out)
        out = self.conv1(out) 
        
        # Block 2: GN -> SiLU -> Conv1D
        out = self.gn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        
        # Residual Connection (Note: Final SiLU is often omitted in the main path)
        out += residual        
        return self.out_act(out)
    
class ResNet1D(nn.Module):
    def __init__(self,
                 in_channels: int = 7,
                 hidden_channels: int = 32,
                 out_channels: int = 1,
                 num_blocks: int = 4,
                 kernel_size: int = 3,
                 num_groups: int = 4):
        super().__init__()

        padding = kernel_size // 2

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size,
                      padding=padding),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels),
            nn.SiLU(),
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_channels,
                                     kernel_size=kernel_size,
                                     num_groups=num_groups))
        self.blocks = nn.Sequential(*blocks)
        self.head   = nn.Conv1d(hidden_channels, out_channels, kernel_size,
                                padding=padding)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        # x: (B, C_in, L)
        out = self.stem(x)
        out = self.blocks(out)
        out = self.head(out)
        return out