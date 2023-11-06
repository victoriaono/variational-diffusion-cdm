import numpy as np
import torch
from torch import einsum, nn, pi, softmax
import torch.nn.functional as F
from .utils import zero_init

class UNetVDM(nn.Module):
    def __init__(
        self,
        n_attention_heads: int = 1,
        embedding_dim: int = 128,
        norm_groups: int = 32,
        dropout_prob: float = 0.1,
        use_fourier_features: bool = True,
        input_channels: int = 1,
        conditioning_channels: int = 1,
        add_attention: bool = False,
        attention_everywhere: bool = False,
        n_blocks: int = 32,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,       
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_fourier_features = use_fourier_features
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.add_attention = add_attention
        
        attention_params = dict(
            n_heads=n_attention_heads,
            n_channels=embedding_dim,
            norm_groups=norm_groups,
        )
        resnet_params = dict(
            ch_in=embedding_dim,
            ch_out=embedding_dim,
            condition_dim=4 * embedding_dim,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=-4.0,
                last=2,
                step=1,
            )
        self.embed_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
        )
        total_input_ch = input_channels + conditioning_channels
        if use_fourier_features:
            total_input_ch *= 1 + self.fourier_features.num_features
        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, kernel_size=3, padding=1,padding_mode="circular")
        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.down_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params)
                if attention_everywhere
                else None,
            )
            for _ in range(n_blocks)
        )

        self.mid_resnet_block_1 = ResnetBlock(**resnet_params)
        
        if self.add_attention:
            self.mid_attn_block = AttentionBlock(**attention_params)
            
        self.mid_resnet_block_2 = ResnetBlock(**resnet_params)

        # Up path: n_blocks+1 blocks with a resnet block and maybe attention.
        resnet_params["ch_in"] *= 2  # double input channels due to skip connections
        self.up_blocks = nn.ModuleList(
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params)
                if attention_everywhere
                else None,
            )
            for _ in range(n_blocks + 1)
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(embedding_dim, input_channels, 3, padding=1,padding_mode="circular")),
        )

    def forward(
        self,
        z,
        g_t,
        conditioning=None,
    ):
        if conditioning is not None:
            z_concat = torch.concat(
                (z, conditioning),
                axis=1,
            )
        else:
            z_concat = z
        # Get gamma to shape (B, ).
        g_t = g_t.expand(z_concat.shape[0])  # assume shape () or (1,) or (B,)
        assert g_t.shape == (z_concat.shape[0],)
        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        g_t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(g_t, self.embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding)
        h = self.maybe_concat_fourier(z_concat)    
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            hs.append(h)
            h = down_block(h, cond)
        hs.append(h)
        h = self.mid_resnet_block_1(h, cond)
        
        if self.add_attention:
            h = self.mid_attn_block(h)
            
        h = self.mid_resnet_block_2(h, cond)
        for up_block in self.up_blocks:  # n_blocks+1 times
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block(h, cond)
        prediction = self.conv_out(h)
        return prediction + z

    def maybe_concat_fourier(self, z):
        if self.use_fourier_features:
            return torch.cat([z, self.fourier_features(z)], dim=1)
        return z

def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    timesteps *= 1000
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)


def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)


class Attention(nn.Module):
    """Based on https://github.com/openai/guided-diffusion."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*H*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, H*C, T)
        return out.view(*out.shape[:2], *spatial_dims)


class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        assert n_channels % n_heads == 0
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,ch_in,ch_out=None,condition_dim=None,dropout_prob=0.0,norm_groups=32,aspp=None):
        super().__init__()
        ch_out = ch_in if ch_out is None else ch_out
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.condition_dim = condition_dim
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            (nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1,padding_mode="circular") if (aspp is None) else ASPPConv2D(ch_in,ch_out,aspp,padding_mode="circular")),
        )
        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            (zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1,padding_mode="circular")) if (aspp is None) else zero_init(ASPPConv2D(ch_out,ch_out,aspp,padding_mode="circular"))),
        )
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, condition):
        h = self.net1(x)
        if condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim)
            condition = self.cond_proj(condition)
            condition = condition[:, :, None, None] #2d
            h = h + condition
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        return x + h

class ASPP2D_CR(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,padding_mode="zeros"):
        super().__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False,padding_mode=padding_mode)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.conv(x))


class ASPPConv2D(nn.Module):
    def __init__(self, in_channels,out_channels, dilations,padding_mode="zeros"):
        super().__init__()
        self.relu=nn.ReLU(inplace=True)

        self.conv=nn.Conv2d(in_channels, out_channels, 1, bias=False)

        DilConvs = []
        for dilation in dilations:
            DilConvs.append(ASPP2D_CR(in_channels=in_channels, out_channels=out_channels, dilation=dilation,padding_mode=padding_mode))

        self.DilConvs = nn.ModuleList(DilConvs)

        self.compressionconv = nn.Conv2d((len(dilations)+1) * out_channels, out_channels, 1, bias=False)

    def forward(self, x):
        res = []
        res.append(self.conv(x))
        for conv in self.DilConvs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.relu(self.compressionconv(res))

class DownBlock(nn.Module):
    def __init__(self, resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.down=nn.Conv2d(self.resnet_block.ch_out,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, cond):
        xskip = self.resnet_block(x, cond)
        x=self.down(xskip)
        return x,xskip

class UpBlock(nn.Module):
    def __init__(self,resnet_block):
        super().__init__()
        self.resnet_block = resnet_block
        self.up=nn.ConvTranspose2d(self.resnet_block.ch_out*2,self.resnet_block.ch_out,2,stride=2)

    def forward(self, x, xskip, cond):
        xu=self.up(x)
        x=torch.cat([xu,xskip],dim=1)
        x = self.resnet_block(x, cond)
        return x

class UNetThin(nn.Module):#n_channels fixed to embedding_dim
    def __init__(
        self,
        input_channels: int = 1,
        conditioning_channels: int = 1,
        embedding_dim: int=32,
        n_blocks: int = 4,  
        norm_groups: int = 8,

        dropout_prob: float = 0.1,
        use_fourier_features: bool = True,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,   
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_fourier_features = use_fourier_features
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        resnet_params = dict(
            ch_in=embedding_dim,
            ch_out=embedding_dim,
            condition_dim=4 * embedding_dim,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=-4.0,
                last=2,
                step=1,
            )
        self.embed_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
        )
        total_input_ch = input_channels + conditioning_channels
        if use_fourier_features:
            total_input_ch *= 1 + self.fourier_features.num_features
        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, kernel_size=3, padding=1,padding_mode="circular")
        
        #Down of UNet
        self.down_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.down_blocks.append(DownBlock(resnet_block=ResnetBlock(**resnet_params)))

        self.mid_resnet_block = ResnetBlock(**resnet_params)

        #Up of UNet
        resnet_params["ch_in"] *= 2  # double input channels due to skip connections
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.up_blocks.append(UpBlock(resnet_block=ResnetBlock(**resnet_params)))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(embedding_dim, input_channels, 3, padding=1,padding_mode="circular")),
        )

    def forward(self,z,g_t,conditioning=None):
        #concatenate conditioning
        if conditioning is not None:
            z_concat = torch.concat((z, conditioning),axis=1,)
        else:
            z_concat = z

        # Get gamma to shape (B, ).
        g_t = g_t.expand(z_concat.shape[0])  # shape () or (1,) or (B,) -> (B,)
        assert g_t.shape == (z_concat.shape[0],)

        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        g_t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(g_t, self.embedding_dim) #(B, embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding) # (B, 4 * embedding_dim)

        h = self.maybe_concat_fourier(z_concat) #(B, C+n_fourier_features, H, W, D)

        #standard UNet from here but with cond at each layer
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            h,hskip = down_block(h, cond=cond)
            hs.append(hskip)
        h = self.mid_resnet_block(h, cond)
        for up_block in self.up_blocks:  # n_blocks times
            h = up_block(x=h,xskip=hs.pop(),cond=cond)
        prediction = self.conv_out(h)
        return prediction + z

    def maybe_concat_fourier(self, z):
        if self.use_fourier_features:
            return torch.cat([z, self.fourier_features(z)], dim=1)
        return z

class UNetThin2(nn.Module):#n_channels increasing by *2
    def __init__(
        self,
        input_channels: int = 1,
        conditioning_channels: int = 1,
        embedding_dim: int=32,
        n_blocks: int = 4,  
        norm_groups: int = 8,
        #dilation: int =1,

        dropout_prob: float = 0.1,
        use_fourier_features: bool = False,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_fourier_features = use_fourier_features
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        resnet_params = dict(
            condition_dim=4 * embedding_dim,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=-4.0,
                last=2,
                step=1,
            )
        self.embed_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
        )
        total_input_ch = input_channels + conditioning_channels
        if use_fourier_features:
            total_input_ch *= 1 + self.fourier_features.num_features
        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, kernel_size=3, padding=1,padding_mode="circular")
        
        #Down of UNet
        self.down_blocks = nn.ModuleList()
        dim=embedding_dim
        for i in range(n_blocks):
            self.down_blocks.append(DownBlock(resnet_block=ResnetBlock(ch_in=(dim//2 if i!=0 else dim),ch_out=dim,**resnet_params)))
            dim*=2

        self.mid_resnet_block = ResnetBlock(ch_in=dim//2,ch_out=dim,**resnet_params)

        #Up of UNet
        self.up_blocks = nn.ModuleList()
        for i in range(n_blocks):
            dim//=2
            self.up_blocks.append(UpBlock(resnet_block=ResnetBlock(ch_in=dim*2,ch_out=dim,**resnet_params)))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            zero_init(nn.Conv2d(embedding_dim, input_channels, 3, padding=1,padding_mode="circular")),
        )

    def forward(self,z,g_t,conditioning=None):
        #concatenate conditioning
        if conditioning is not None:
            z_concat = torch.concat((z, conditioning),axis=1,)
        else:
            z_concat = z

        # Get gamma to shape (B, ).
        g_t = g_t.expand(z_concat.shape[0])  # shape () or (1,) or (B,) -> (B,)
        assert g_t.shape == (z_concat.shape[0],)

        # Rescale to [0, 1], but only approximately since gamma0 & gamma1 are not fixed.
        g_t = (g_t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_embedding = get_timestep_embedding(g_t, self.embedding_dim) #(B, embedding_dim)
        # We will condition on time embedding.
        cond = self.embed_conditioning(t_embedding) # (B, 4 * embedding_dim)

        h = self.maybe_concat_fourier(z_concat) #(B, C+n_fourier_features, H, W, D)

        #standard UNet from here but with cond at each layer
        h = self.conv_in(h)  # (B, embedding_dim, H, W)
        #print(h.shape)
        hs = []
        for down_block in self.down_blocks:  # n_blocks times
            h,hskip = down_block(h, cond=cond)
            hs.append(hskip)
            #print(h.shape)
        h = self.mid_resnet_block(h, cond)
        #print(h.shape)
        for up_block in self.up_blocks:  # n_blocks times
            h = up_block(x=h,xskip=hs.pop(),cond=cond)
            #print(h.shape)
        prediction = self.conv_out(h)
        return prediction + z

    def maybe_concat_fourier(self, z):
        if self.use_fourier_features:
            return torch.cat([z, self.fourier_features(z)], dim=1)
        return z
