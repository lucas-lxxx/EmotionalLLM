"""三种语音畸变方法实现

参考 Gao et al. "Black-box adversarial attacks through speech distortion
for speech emotion recognition":
  1. VTLN (Vocal Tract Length Normalization) — 频率扭曲
  2. McAdams transformation — 共振峰频率修改
  3. MSS (Modulation Spectrum Smoothing) — 调制谱平滑
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, butter


def vtln(waveform: np.ndarray, sr: int, alpha: float = 0.15,
         n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Vocal Tract Length Normalization (VTLN)

    通过频率扭曲函数修改幅度谱，再用原始相位 ISTFT 重建。

    Args:
        waveform: (samples,) mono audio, float
        sr: sample rate
        alpha: warping factor ∈ [-1, 1], 0 = no warping
        n_fft: FFT size
        hop_length: hop length for STFT

    Returns:
        warped waveform, same length as input
    """
    if abs(alpha) < 1e-6:
        return waveform.copy()

    # STFT
    from scipy.signal import stft as scipy_stft, istft as scipy_istft
    f, t, Zxx = scipy_stft(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    n_freq = magnitude.shape[0]

    # 频率扭曲映射
    # ω1 = π*ω0 + 2*arctan(alpha*sin(π*ω) / (1 - alpha*cos(π*ω)))
    freqs_norm = np.linspace(0, 1, n_freq)  # normalized [0, 1]
    warped_freqs = np.zeros_like(freqs_norm)
    for i, w in enumerate(freqs_norm):
        warped_freqs[i] = np.pi * w + 2 * np.arctan2(
            alpha * np.sin(np.pi * w),
            1 - alpha * np.cos(np.pi * w)
        )
    warped_freqs = warped_freqs / np.pi  # back to [0, ~1]
    warped_freqs = np.clip(warped_freqs, 0, 1)

    # 将扭曲后的频率映射到原始频率 bin 索引
    warped_indices = warped_freqs * (n_freq - 1)

    # 插值生成新的幅度谱
    warped_magnitude = np.zeros_like(magnitude)
    for t_idx in range(magnitude.shape[1]):
        warped_magnitude[:, t_idx] = np.interp(
            np.arange(n_freq),
            warped_indices,
            magnitude[:, t_idx],
        )

    # 用扭曲后的幅度 + 原始相位重建
    Zxx_warped = warped_magnitude * np.exp(1j * phase)
    _, x_warped = scipy_istft(Zxx_warped, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # 对齐长度
    orig_len = len(waveform)
    if len(x_warped) > orig_len:
        x_warped = x_warped[:orig_len]
    elif len(x_warped) < orig_len:
        x_warped = np.pad(x_warped, (0, orig_len - len(x_warped)))

    return x_warped.astype(np.float32)


def mcadams(waveform: np.ndarray, sr: int, alpha: float = 0.80,
            lpc_order: int = 20, frame_len: int = 640,
            hop_length: int = 160) -> np.ndarray:
    """McAdams transformation

    通过 LPC 分析修改共振峰频率（极点角度取 alpha 次幂），再重建语音。

    Args:
        waveform: (samples,) mono audio, float
        sr: sample rate
        alpha: McAdams coefficient, α>1 扩展频率, α<1 压缩频率
        lpc_order: LPC analysis order
        frame_len: analysis frame length in samples
        hop_length: hop between frames

    Returns:
        transformed waveform
    """
    if abs(alpha - 1.0) < 1e-6:
        return waveform.copy()

    from scipy.signal import lfilter
    from numpy.polynomial import polynomial as P

    n_samples = len(waveform)
    output = np.zeros(n_samples, dtype=np.float32)
    window = np.hanning(frame_len)

    for start in range(0, n_samples - frame_len + 1, hop_length):
        frame = waveform[start:start + frame_len] * window

        # LPC analysis
        try:
            from scipy.linalg import toeplitz
            # Autocorrelation method
            r = np.correlate(frame, frame, mode='full')
            r = r[len(frame) - 1:]  # positive lags
            r = r[:lpc_order + 1]

            # Levinson-Durbin
            a = _levinson_durbin(r, lpc_order)
            if a is None:
                output[start:start + frame_len] += frame
                continue
        except Exception:
            output[start:start + frame_len] += frame
            continue

        # 求 LPC 多项式的根（极点）
        roots = np.roots(a)

        # 修改极点角度
        new_roots = []
        for root in roots:
            mag = np.abs(root)
            angle = np.angle(root)
            if np.abs(angle) > 1e-6:  # 非零虚部的极点
                # 角度取 alpha 次幂（保持符号）
                new_angle = np.sign(angle) * (np.abs(angle) ** alpha)
                new_root = mag * np.exp(1j * new_angle)
            else:
                new_root = root
            new_roots.append(new_root)

        new_roots = np.array(new_roots)

        # 从新极点重建 LPC 系数
        new_a = np.real(np.poly(new_roots))

        # 用原始残差 + 新 LPC 合成
        residual = lfilter(a, [1.0], frame)
        synth_frame = lfilter([1.0], new_a, residual)

        output[start:start + frame_len] += synth_frame.astype(np.float32) * window

    # 归一化 overlap-add 的窗函数
    norm = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples - frame_len + 1, hop_length):
        norm[start:start + frame_len] += window ** 2
    norm = np.maximum(norm, 1e-8)
    output /= norm

    return output


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray | None:
    """Levinson-Durbin recursion for LPC coefficients."""
    if r[0] == 0:
        return None

    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        lam = 0.0
        for j in range(1, i):
            lam += a[j] * r[i - j]
        lam = (r[i] - lam) / e

        # Update coefficients
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] - lam * a[i - j]
        a_new[i] = -lam
        a = a_new

        e = e * (1 - lam ** 2)
        if e <= 0:
            return None

    return a


def mss(waveform: np.ndarray, sr: int, alpha: float = 0.25,
        n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Modulation Spectrum Smoothing (MSS)

    对每个频率 bin 的时间序列（log 幅度）进行低通滤波，
    去除快速的时间调制，保留慢变化。

    Args:
        waveform: (samples,) mono audio, float
        sr: sample rate
        alpha: cutoff frequency (normalized, ∈ [0, 1]),
               smaller = more smoothing
        n_fft: FFT size
        hop_length: hop length

    Returns:
        smoothed waveform
    """
    from scipy.signal import stft as scipy_stft, istft as scipy_istft

    f, t, Zxx = scipy_stft(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # 对每个频率 bin 的 log 幅度做低通滤波
    log_mag = np.log(magnitude + 1e-8)

    # 设计低通滤波器
    cutoff = max(alpha, 0.01)  # 避免 cutoff=0
    cutoff = min(cutoff, 0.99)
    b, a = butter(4, cutoff, btype='low')

    smoothed_log_mag = np.zeros_like(log_mag)
    for f_idx in range(log_mag.shape[0]):
        if log_mag.shape[1] > 12:  # 需要足够长的序列
            try:
                smoothed_log_mag[f_idx, :] = lfilter(b, a, log_mag[f_idx, :])
            except Exception:
                smoothed_log_mag[f_idx, :] = log_mag[f_idx, :]
        else:
            smoothed_log_mag[f_idx, :] = log_mag[f_idx, :]

    smoothed_magnitude = np.exp(smoothed_log_mag)

    # 用平滑后的幅度 + 原始相位重建
    Zxx_smoothed = smoothed_magnitude * np.exp(1j * phase)
    _, x_smoothed = scipy_istft(Zxx_smoothed, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # 对齐长度
    orig_len = len(waveform)
    if len(x_smoothed) > orig_len:
        x_smoothed = x_smoothed[:orig_len]
    elif len(x_smoothed) < orig_len:
        x_smoothed = np.pad(x_smoothed, (0, orig_len - len(x_smoothed)))

    return x_smoothed.astype(np.float32)
