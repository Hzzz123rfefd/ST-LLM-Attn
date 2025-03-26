import torch.nn as nn


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