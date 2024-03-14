import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from data.constants import norms, boxsize

def kl_std_normal(mean_squared, var):
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)


class FixedLinearSchedule(torch.nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t


class LearnedLinearSchedule(torch.nn.Module):
    def __init__(self, gamma_min, gamma_max,gamma_min_max=None):
        super().__init__()
        self.gamma_min_max=gamma_min_max
        self.b = torch.nn.Parameter(torch.tensor(gamma_min))
        self.w = torch.nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        if self.gamma_min_max is None:
            return self.b + self.w.abs() * t
        else:
            return torch.clamp(self.b,min=None,max=self.gamma_min_max) + self.w.abs() * t


@torch.no_grad()
def zero_init(module: torch.nn.Module) -> torch.nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module


def power(x,x2=None):
    """
    Parameters
    ---------------------
    x: the input field, in torch tensor
    
    x2: the second field for cross correlations, if set None, then just compute the auto-correlation of x
    
    ---------------------
    Compute power spectra of input fields
    Each field should have batch and channel dimensions followed by spatial
    dimensions. Powers are summed over channels, and averaged over batches.

    Power is not normalized. Wavevectors are in unit of the fundamental
    frequency of the input.
    
    source code adapted from 
    https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTBase
    """
    signal_ndim = x.dim() - 2
    signal_size = x.shape[-signal_ndim:]
    
    kmax = min(s for s in signal_size) // 2
    even = x.shape[-1] % 2 == 0
    
    x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
    if x2 is None:
        x2 = x
    else:
        x2 = torch.fft.rfftn(x2, s=signal_size)
    P = x * x2.conj()
    
    P = P.mean(dim=0)
    P = P.sum(dim=0)
    
    del x, x2
    
    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape]
    k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
    k = torch.meshgrid(*k,indexing="ij")
    k = torch.stack(k, dim=0)
    k = k.norm(p=2, dim=0)

    N = torch.full_like(P, 2, dtype=torch.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1

    k = k.flatten().real
    P = P.flatten().real
    N = N.flatten().real

    kbin = k.ceil().to(torch.int32)
    k = torch.bincount(kbin, weights=k * N)
    P = torch.bincount(kbin, weights=P * N)
    N = torch.bincount(kbin, weights=N).round().to(torch.int32)
    del kbin

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1:1+kmax]
    P = P[1:1+kmax]
    N = N[1:1+kmax]

    k /= N
    P /= N

    return k, P, N

def pk(fields):
    kss,pkss,nss = [],[],[]
    for field in fields:
        ks,pks,ns = power(field[None])#add 1 batch
        kss.append(ks)
        pkss.append(pks)
        nss.append(ns)
    return torch.stack(kss,dim=0),torch.stack(pkss,dim=0),torch.stack(nss,dim=0)

k_conversion = 2*np.pi/boxsize
def compute_pk(field, field_b=None,):
    # Assumes field has shape (1,1,Npixels, Npixels)
    assert len(field.shape) == 4
    if field_b is not None:
        assert len(field_b.shape) == 4
        k, pk, _ = power(
            torch.Tensor(field/np.sum(field)),
            torch.Tensor(field_b/np.sum(field_b)),
        )
    else:
        k, pk, _ = power(
            torch.Tensor(field/np.sum(field)),
        )
    k *= k_conversion
    pk *= boxsize**2
    return k, pk

class MonotonicLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.weight.abs(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class NNSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max,mid_dim=1024):
        super().__init__()
        self.mid_dim=mid_dim
        self.l1=MonotonicLinear(1,1,bias=True)
        self.l1.weight.data[0,0]=gamma_max-gamma_min
        self.l1.bias.data[0]=gamma_min
        self.l2=MonotonicLinear(1,self.mid_dim,bias=True)
        self.l3=MonotonicLinear(self.mid_dim,1,bias=False)

    def forward(self, t):
        t_sh=t.shape
        t=t.reshape(-1,1)
        g=self.l1(t)
        _g=2.*(t-0.5)
        _g=self.l2(_g)
        _g=2.*(torch.sigmoid(_g)-0.5)
        _g=self.l3(_g)/self.mid_dim
        g=g+_g
        return g.reshape(t_sh)

def draw_figure(x,sample,conditioning,dataset):
    mean_input = norms[dataset][0]
    std_input= norms[dataset][1]
    mean_target= norms[dataset][2]
    std_target= norms[dataset][3]
    fontsize = 14
    fig, ax = plt.subplots(2,3,figsize=(15,10))            
    ax.flat[0].imshow(conditioning[0].squeeze().cpu(), cmap='copper', vmin=-.1, vmax=11)
    ax.flat[1].imshow(x[0].squeeze().cpu(), cmap='cividis', vmin=-2, vmax=6)
    ax.flat[2].imshow(sample[0].squeeze().cpu(), cmap='cividis', vmin=-2, vmax=6)
    ax.flat[0].set_title("Stars", fontsize=fontsize)
    ax.flat[1].set_title("True DM", fontsize=fontsize)
    ax.flat[2].set_title("Sampled DM", fontsize=fontsize)

    ax.flat[3].hist(x[0].cpu().numpy().flatten(), bins=np.linspace(-4, 4, 50), histtype='step', color='#4c4173', label="True DM", density=True)
    ax.flat[3].hist(sample[0].cpu().numpy().flatten(), bins=np.linspace(-4, 4, 50), alpha=0.5, color='#4c4173', label="Sampled DM", density=True)
    ax.flat[3].legend(fontsize=9)
    ax.flat[3].set_title("Density", fontsize=fontsize)

    conditioning = 10 ** (conditioning * std_input + mean_input)
    k, P = compute_pk(conditioning.cpu().numpy())
    ax.flat[4].loglog(k, P, label="Stars", color='#c09465')
    x = 10 ** (x * std_target + mean_target)
    k, P = compute_pk(x.cpu().numpy())
    ax.flat[4].loglog(k, P, label="True DM", color='#4c4173')
    sample = 10 ** (sample * std_target + mean_target)
    k, P = compute_pk(sample.cpu().numpy())
    ax.flat[4].loglog(k, P, label="Sampled DM", color='#709bb5')
    ax.flat[4].legend(fontsize=9)
    ax.flat[4].set_xlabel('k',fontsize=fontsize)
    ax.flat[4].set_ylabel('P(k)',fontsize=fontsize)
    ax.flat[4].set_title("Power Spectrum", fontsize=fontsize)
    mcdm_gen = sample.cpu().numpy()
    nmaps = mcdm_gen.shape[0]
    flattened_sample = np.average(mcdm_gen.reshape(nmaps,-1),axis=-1)
    flattened_x = np.average(x.cpu().numpy().reshape(nmaps,-1),axis=-1)
    ax.flat[5].scatter(flattened_x, flattened_sample)
    ax.flat[5].plot(flattened_x, flattened_x, color='grey')
    return fig