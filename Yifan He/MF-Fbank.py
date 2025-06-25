# -*- coding: utf-8 -*-

import math
from typing import Callable, Optional
from warnings import warn

import torch
from torch import Tensor
from torchaudio import functional as F
from torchaudio.compliance import kaldi
import numpy as np

class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return F.spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length,
                             self.win_length, self.power, self.normalized)

    
class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``None``)
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: Optional[int] = None) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)

        fb = torch.empty(0) if n_stft is None else F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate)
        self.register_buffer('fb', fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

        return mel_specgram




    
class MF_Fbank(torch.nn.Module):
    
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min', "t", "window", "a"]

    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 pad: int = 0,
                 n_mels: int = 128,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 epoch: int = 999,
                 wkwargs: Optional[dict] = None) -> None:
        super(MF_Fbank, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.epoch = epoch
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=self.power,
                                       normalized=self.normalized, wkwargs=wkwargs)
        self.mel_scale = MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1)        
        # 定义每个特征的自适应权重（初始化为1）
        self.raw_weight_a = torch.nn.Parameter(torch.tensor(1.0))
        self.raw_weight_b = torch.nn.Parameter(torch.tensor(1.0))
        self.raw_weight_c = torch.nn.Parameter(torch.tensor(1.0))
        self.raw_weight_d = torch.nn.Parameter(torch.tensor(1.0))
        self.raw_weight_e = torch.nn.Parameter(torch.tensor(1.0))        
        # 权重是否可更新的标志
        self.freeze_weights = False
        # 初始化 fixed_weights 为 nn.Parameter
        self.fixed_weights = torch.nn.Parameter(torch.ones(5), requires_grad=False)  # 初始化为 1，设置为不可训练
    def forward(self, waveform: Tensor) -> Tensor:
        # 生成时间序列，确保时间点的数量与窗口函数的长度相同
        t = torch.linspace(0, 1, self.win_length)
        window = torch.hamming_window(self.win_length).to(waveform.device)
        a = [1,0.8,0.6,0.4,0.2]
        specgram_list = []
        for i in a:
            if i == 1 :
                specgram = F.spectrogram(waveform, self.pad, window, self.n_fft, self.hop_length,
                                         self.win_length, self.power, self.normalized)
            if 0 < i < 1 :
                p = math.radians(90 * i)
                q = 1/math.tan(p)
                phase = q * t**2 / 2   # 实部相位
                # 创建一个“纯虚”张量，再和“纯实”拼起来
                chirp_signal = torch.complex(torch.zeros_like(phase), phase).exp()  # e^{j*phase}
                fractional_window = chirp_signal * window
                fractional_window_real = fractional_window.real
                fractional_window_imag = fractional_window.imag
                specgram_real = F.spectrogram(waveform, self.pad, fractional_window_real, self.n_fft, self.hop_length,
                                         self.win_length, self.power, self.normalized)
                specgram_imag = F.spectrogram(waveform, self.pad, fractional_window_imag, self.n_fft, self.hop_length,
                                              self.win_length, self.power, self.normalized)
                specgram = torch.abs(specgram_real + 1j * specgram_imag)
            mel_specgram = self.mel_scale(specgram)
            specgram_list.append(mel_specgram)
        xa = specgram_list[0] + 1e-6
        xb = specgram_list[1] + 1e-6
        xc = specgram_list[2] + 1e-6
        xd = specgram_list[3] + 1e-6
        xe = specgram_list[4] + 1e-6           
        xa = xa.log()
        xb = xb.log()
        xc = xc.log()
        xd = xd.log()
        xe = xe.log()
        xa = xa - torch.mean(xa, dim=-1, keepdim=True)
        xb = xb - torch.mean(xb, dim=-1, keepdim=True)
        xc = xc - torch.mean(xc, dim=-1, keepdim=True)
        xd = xd - torch.mean(xd, dim=-1, keepdim=True)
        xe = xe - torch.mean(xe, dim=-1, keepdim=True)

        if self.epoch <= 20:  # 前10%训练时间内权重可变
            self.freeze_weights = False
        else:
            self.freeze_weights = True

        if not self.freeze_weights:
            raw_weights = torch.stack([self.raw_weight_a, self.raw_weight_b, self.raw_weight_c, self.raw_weight_d, self.raw_weight_e])
            weights = torch.softmax(raw_weights, dim=0) * 5
            self.fixed_weights.data = weights  # 用 data 来更新值而不影响梯度计算 
        else:
            # 使用第20个epoch后的固定权重
            weights = self.fixed_weights

        weight_a, weight_b, weight_c, weight_d, weight_e = weights

#         # 使用可训练的权重调整每段特征的大小
        xa = xa * weight_a
        xb = xb * weight_b
        xc = xc * weight_c
        xd = xd * weight_d
        xe = xe * weight_e
        
        if not self.freeze_weights:        
#             # 打印权重
        print(f"Weight A: {weight_a.item():.4f}, Weight B: {weight_b.item():.4f}, "
              f"Weight C: {weight_c.item():.4f}, Weight D: {weight_d.item():.4f}, "
              f"Weight E: {weight_e.item():.4f}")
        
        x = torch.cat((xa,xb,xc,xd,xe), dim=2)
        return x

