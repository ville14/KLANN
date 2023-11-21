import math
import torch
from torch import nn
from torchaudio import functional
import numpy as np

# state variable filter (SVF) trained in frequency domain and inferred in time domain
class DSVF(nn.Module):
    def __init__(self, N):
        super().__init__()
        # filter parameters
        self.g = nn.Parameter(torch.zeros(1))
        self.R = nn.Parameter(torch.zeros(1))
        self.m_hp = nn.Parameter(torch.ones(1))
        self.m_bp = nn.Parameter(torch.ones(1))
        self.m_lp = nn.Parameter(torch.ones(1))

        # parameters for the STFT and the overlap-add method
        self.N = N
        self.nfft = 2 ** math.ceil(math.log2(2*self.N-1))

    # x.shape -> (batch_size, time)
    def forward(self, x):
        # restrict frequencies to (0, pi)
        g = torch.tan(torch.pi*1/(1+torch.exp(-self.g))/2)
        # restrict R to > 0
        R = nn.functional.softplus(self.R)
        g_2 = g * g
        b = torch.cat((g_2 * self.m_lp + g * self.m_bp + self.m_hp,
                       2 * g_2 * self.m_lp - 2 * self.m_hp,
                       g_2 * self.m_lp - g * self.m_bp + self.m_hp), dim=0)
        a = torch.cat((g_2 + 2 * R * g + 1,
                       2 * g_2 - 2,
                       g_2 - 2 * R * g + 1), dim=0)

        # filter in frequency domain
        if self.training:
            # divide x into sub-frames of length N and perform convolution in frequency domain
            # (length of x must be divisible by N)
            segments = x.view(x.shape[0], -1, self.N)
            X = torch.fft.rfft(segments, n = self.nfft, dim=-1)
            H = torch.fft.rfft(b, n = self.nfft, dim=-1) / torch.fft.rfft(a, n = self.nfft, dim=-1)
            y = torch.fft.irfft(X * H, n = self.nfft, dim=-1)

            if segments.shape[1] == 1:
                return y[:,:,0:self.N].flatten(-2)
            else:
            # overlap-add
                firstPart = y[:,:,0:self.N]
                overlap = y[:,:-1,self.N:2*self.N]
                overlapExt = nn.functional.pad(overlap, (0,0,1,0), "constant", 0) # pad the first frame
                return (firstPart + overlapExt).flatten(-2)

        # filter in time domain
        else:
            return functional.lfilter(x, a, b, clamp = False)

# DSVFs in parallel
class MODEL1(nn.Module):
    def __init__(self, layers, n, N):
        super().__init__()
        self.n = n
        mlp1 = []
        mlp1.append(nn.Linear(1, 2*layers[0]))
        mlp1.append(nn.GLU())
        for i in range(1, len(layers)):
            mlp1.append(nn.Linear(layers[i-1], 2*layers[i]))
            mlp1.append(nn.GLU())
        mlp1.append(nn.Linear(layers[-1], n))
        self.mlp1 = nn.Sequential(*mlp1)

        self.filters =  nn.ModuleList([])
        for _ in range(self.n):
            self.filters.append(DSVF(N))

        layers.reverse()
        mlp2 = []
        mlp2.append(nn.Linear(n, 2*layers[0]))
        mlp2.append(nn.GLU())
        for i in range(1, len(layers)):
            mlp2.append(nn.Linear(layers[i-1], 2*layers[i]))
            mlp2.append(nn.GLU())
        mlp2.append(nn.Linear(layers[-1], 1))
        self.mlp2 = nn.Sequential(*mlp2)

    def forward(self, x):
        z = self.mlp1(x)
        y = []
        for i in range(self.n):
            y.append(self.filters[i](z[:,:,i]).unsqueeze(-1))
        return self.mlp2(torch.cat(y, dim=-1))

# DSVFs in parallel and series
class MODEL2(nn.Module):
    def __init__(self, layers, layer, n, N):
        super().__init__()
        self.n = n
        mlp1 = []
        mlp1.append(nn.Linear(1, 2*layers[0]))
        mlp1.append(nn.GLU())
        for i in range(1, len(layers)):
            mlp1.append(nn.Linear(layers[i-1], 2*layers[i]))
            mlp1.append(nn.GLU())
        mlp1.append(nn.Linear(layers[-1], n))
        self.mlp1 = nn.Sequential(*mlp1)

        self.linear =  nn.ModuleList([])
        for _ in range(self.n-1):
            self.linear.append(nn.Sequential(
                nn.Linear(2, 2*layer),
                nn.GLU(),
                nn.Linear(layer, 1)
            ))
        self.filter =  nn.ModuleList([])
        for _ in range(self.n):
            self.filter.append(DSVF(N = N))

        layers.reverse()
        mlp2 = []
        mlp2.append(nn.Linear(n, 2*layers[0]))
        mlp2.append(nn.GLU())
        for i in range(1, len(layers)):
            mlp2.append(nn.Linear(layers[i-1], 2*layers[i]))
            mlp2.append(nn.GLU())
        mlp2.append(nn.Linear(layers[-1], 1))
        self.mlp2 = nn.Sequential(*mlp2)

    # x -> (batch_size, samples, input_size)
    def forward(self, x):
        y = self.mlp1(x)
        z = self.filter[0](y[:,:,0]).unsqueeze(-1)
        z_s = []
        z_s.append(z)
        for i in range(self.n-1):
            z = self.filter[i+1](self.linear[i](torch.cat((z, y[:,:,i+1].unsqueeze(-1)), dim=-1)).squeeze(-1)).unsqueeze(-1)
            z_s.append(z)
        return self.mlp2(torch.cat(z_s, dim=-1))
    # return -> (batch_size, samples, input_size)
