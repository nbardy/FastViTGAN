#
# Is a model architecture that is used to generate images.
# It uses a codebook to project clip embeddings to the
#
# It it based on MobileStyleGAN(https://arxiv.org/abs/2104.04767)
#
# The decoder is named MobileSynthesisNetwork
import math
import torch.nn.functional as F
import torch
import torch.nn as nn


from pytorch_wavelets.dwt.lowlevel import *
import torch
import torch.nn as nn


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight = nn.Parameter(
            torch.randn(channels_out, channels_in, kernel_size, kernel_size)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer(
                "style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight, padding=self.padding)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = self.weight.unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv *
                           w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)


class ModulatedDWConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight_dw = nn.Parameter(
            torch.randn(channels_in, 1, kernel_size, kernel_size)
        )
        self.weight_permute = nn.Parameter(
            torch.randn(channels_out, channels_in, 1, 1)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer(
                "style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv *
                           w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)


class IDWTUpsampler(nn.Module):
    def __init__(
            self,
            channels_in,
            style_dim,
    ):
        super().__init__()
        self.channels = channels_in // 4
        assert self.channels * 4 == channels_in
        # upsample
        self.idwt = DWTInverse(mode='zero', wave='db1')
        # modulation
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)

    def forward(self, x, style):
        b, _, h, w = x.size()
        x = self.modulation(style).view(b, -1, 1, 1) * x
        low = x[:, :self.channels]
        high = x[:, self.channels:]
        high = high.view(b, self.channels, 3, h, w)
        x = self.idwt((low, [high]))
        return x


class MultichannelImage(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=1
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            channels_in, channels_out, style_dim, kernel_size, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))

    def forward(self, hidden, style):
        out = self.conv(hidden, style)
        out = out + self.bias
        return out


class MobileSynthesisBlock(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedConv2d,
            w_size=3,
    ):
        super().__init__()
        self.w_size = w_size
        self.up = IDWTUpsampler(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.to_img = MultichannelImage(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim ==
                            2 else style[:, 0, :], noise=noise[0])
        hidden = self.conv2(hidden, style if style.ndim ==
                            2 else style[:, 1, :], noise=noise[1])
        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img


# Fast Generator network from MobileStyleGAN
class MobileSynthesisNetwork(nn.Module):
    def __init__(
            self,
            style_dim,
            channels=[512, 512, 512, 512, 512, 256, 128, 64],
            block_w_size=3
    ):
        super().__init__()
        self.style_dim = style_dim

        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv2d(
            channels[0],
            channels[0],
            style_dim,
            kernel_size=3
        )
        self.to_img1 = MultichannelImage(
            channels_in=channels[0],
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module=ModulatedDWConv2d
                )
            )
            channels_in = channels_out

        self.idwt = DWTInverse(mode="zero", wave="db1")
        self.register_buffer("device_info", torch.zeros(1))
        self.trace_model = False

    def forward(self, style, noise=None):
        out = {"noise": [], "freq": [], "img": None}
        noise = NoiseManager(noise, self.device_info.device, self.trace_model)

        hidden = self.input(style)
        out["noise"].append(noise(hidden.size(-1)))
        hidden = self.conv1(hidden, style if style.ndim ==
                            2 else style[:, 0, :], noise=out["noise"][-1])
        img = self.to_img1(hidden, style if style.ndim ==
                           2 else style[:, 1, :])
        out["freq"].append(img)

        for i, m in enumerate(self.layers):
            out["noise"].append(noise(2 ** (i + 3), 2))
            _style = style if style.ndim == 2 else style[:, m.wsize(
            ) * i + 1: m.wsize() * i + m.wsize() + 1, :]
            hidden, freq = m(hidden, _style, noise=out["noise"][-1])
            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])
        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class StyledConv2d(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size,
        demodulate=True,
        conv_module=ModulatedConv2d
    ):
        super().__init__()

        self.conv = conv_module(
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=demodulate
        )

        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.act(out + self.bias)
        return


def _SFB2D(low, highs, g0_row, g1_row, g0_col, g1_col, mode):
    mode = int_to_mode(mode)

    lh, hl, hh = torch.unbind(highs, dim=2)
    lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image
    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """

    def __init__(self, wave='db1', mode='zero', trace_model=False):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.mode = mode
        self.trace_model = trace_model

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            if not self.trace_model:
                ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col,
                                 self.g0_row, self.g1_row, mode)
            else:
                ll = _SFB2D(ll, h, self.g0_col, self.g1_col,
                            self.g0_row, self.g1_row, mode)
        return ll


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.trace_model = False

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        if not hasattr(self, "noise") and self.trace_model:
            self.register_buffer("noise", noise)
        if self.trace_model:
            noise = self.noise
        return image + self.weight * noise


class NoiseManager:
    def __init__(self, noise, device, trace_model=False):
        self.device = device
        self.noise_lut = {}
        if noise is not None:
            for i in range(len(noise)):
                if not None in noise:
                    self.noise_lut[noise[i].size(-1)] = noise[i]
        self.trace_model = trace_model

    def __call__(self, size, b=1):
        if self.trace_model:
            return None if b == 1 else [None] * b
        if size in self.noise_lut:
            return self.noise_lut[size]
        else:
            return torch.randn(b, 1, size, size).to(self.device)
