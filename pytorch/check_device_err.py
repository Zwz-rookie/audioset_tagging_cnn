import torch


def test_rfft_consistency():
    x_cpu = torch.randn(2, 1024, device='cpu', dtype=torch.float32)
    x_cuda = x_cpu.cuda()

    # 自定义实现
    spectrum_cpu = replace_rfft(x_cpu)
    spectrum_cuda = replace_rfft(x_cuda).cpu()

    # PyTorch官方实现
    spectrum_pt = torch.fft.rfft(x_cpu).abs()

    # 误差分析
    print("CPU vs CUDA:", torch.allclose(spectrum_cpu, spectrum_cuda, atol=1e-5))
    print("CPU vs PyTorch:", torch.allclose(spectrum_cpu, spectrum_pt, atol=1e-4))


def replace_rfft(x, n_fft=None):
    # x: [B, T] 或更高维
    if n_fft is None:
        n_fft = x.shape[-1]
    device = x.device
    dtype = x.dtype

    # 构建 DFT matrix（实数和虚数部分）
    freqs = torch.arange(n_fft, device=device).unsqueeze(0)
    idx = torch.arange(n_fft // 2 + 1, device=device).unsqueeze(1)
    angle = 2 * torch.pi * idx * freqs / n_fft

    real_kernels = torch.cos(angle).to(dtype)
    imag_kernels = -torch.sin(angle).to(dtype)

    real = torch.matmul(x, real_kernels.T)
    imag = torch.matmul(x, imag_kernels.T)

    spectrum = torch.sqrt(real ** 2 + imag ** 2)
    return spectrum


if __name__ == '__main__':
    test_rfft_consistency()
