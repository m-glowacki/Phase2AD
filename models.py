
import torch.nn as nn
import torch
import math

def get_positional_encoding(d_model, max_len):
    """
    Generate sinusoidal positional encoding.
    """
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.requires_grad_(False)
    return encodings


class MaskedBatchNorm1D(nn.Module):
    def __init__(self, num_features):
        super(MaskedBatchNorm1D, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, num_features)
        batch_size, seq_len, num_features = x.size()
        mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x_flat = x.view(-1, num_features)  # (batch_size * seq_len, num_features)
        mask_flat = mask.view(-1)          # (batch_size * seq_len)
        # Only consider valid positions
        valid_indices = mask_flat > 0
        if valid_indices.any():
            x_valid = x_flat[valid_indices]
            x_norm = self.bn(x_valid)
            x_flat[valid_indices] = x_norm
        x = x_flat.view(batch_size, seq_len, num_features)
        return x
    

class DeepSetLayer(nn.Module):
    def __init__(self, 
                 d_model=64, 
                 d_ff=128,
                 dropout=0.0):
        super(DeepSetLayer, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.ff = nn.Sequential(
            nn.Linear(d_ff, d_model, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.norm = MaskedBatchNorm1D(d_model)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len)
        weights = (mask.float() / (mask.sum(1, keepdim=True) + 1e-8))[:, :, None]

        x = self.norm(x, mask)
        phi = self.phi(x)
        rho = self.rho((x * weights).sum(1, keepdim=True))
        z = self.ff(rho + phi)
        return z
    
class Encoder(nn.Module):
    def __init__(self, 
                 d_kin=3,
                 d_pid=3,
                 use_pid=True,
                 d_model=64,
                 d_ff=128,
                 d_latent=16,
                 n_layers=4,
                 dropout=0.0):
        super(Encoder, self).__init__()

        d_input = d_kin + d_pid if use_pid else d_kin
        self.ff_input = nn.Linear(d_input, d_model, bias=True)
        self.encoder_layers = nn.ModuleList([
            DeepSetLayer(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.fc_mu = nn.Linear(d_model, d_latent, bias=True)
        self.fc_logvar = nn.Linear(d_model, d_latent, bias=True)

    def forward(self, kin, pid, mask):
        x = torch.cat([kin, pid], dim=-1) if pid is not None else kin
        x = self.ff_input(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        weights = (mask.float() / (mask.sum(1, keepdim=True) + 1e-8))[:, :, None]
        x = (x * weights).sum(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar



class Decoder(nn.Module):
    def __init__(self, 
                 d_pid=8,
                 use_pid=True,
                 d_latent=16,
                 d_model=64,
                 d_ff=128,
                 d_output=3,
                 n_layers=4,
                 dropout=0.0,
                 max_length=100):
        super(Decoder, self).__init__()
        self.d_output = d_output
        self.max_length = max_length
        self.d_latent = d_latent
        self.d_model = d_model
        self.use_pid = use_pid
        self.d_pid = d_pid

        # Positional encodings
        self.register_buffer(
            'positional_encodings',
            get_positional_encoding(d_model, max_length)
        )

        self.ff_input = nn.Linear(d_latent, d_model, bias=True)
        if use_pid:
            self.pid_embedding = nn.Embedding(d_pid, d_model)

        self.decoder_layers = nn.ModuleList([
            DeepSetLayer(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.ff_output = nn.Linear(d_model, d_output, bias=True)

    def forward(self, z, pid, mask):  
        
        z = z.unsqueeze(1)
        x = self.ff_input(z)
        
        if self.use_pid:
            x = x + self.pid_embedding(pid.argmax(-1))
            x = x + self.positional_encodings[
            torch.amax(torch.cumsum(pid, dim = 1) - 1, dim = -1).long()
            ]
        else:
            pos_embedding = self.positional_encodings[:mask.shape[1]].unsqueeze(0)
            x = x + self.positional_encodings[:mask.shape[1]].unsqueeze(0)
        
        for layer in self.decoder_layers:
            x = layer(x, mask)
            
        x = self.ff_output(x)
        return x
