import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding]  # 保持因果性

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        tanh_out = torch.tanh(self.filter_conv(x))
        sigm_out = torch.sigmoid(self.gate_conv(x))
        gated = tanh_out * sigm_out
        residual = self.residual_conv(gated)
        skip = self.skip_conv(gated)
        return x + residual, skip

class STAtt(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=6):
        super(STAtt, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = hidden_dim, nhead = num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        """
        x: (b, t, d) -> (b, t, hidden_dim)
        """
        x = self.fc(x)  
        x = self.transformer_encoder(x) 
        return x