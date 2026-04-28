import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

# 常用命令（PowerShell，可直接复制粘贴）:
# 1) 使用默认参数，处理 TestData 全目录
# python utils/visualize_preprocess_features.py
#
# 2) 使用 3500-8000 Hz 频段，输出到 feature_visualizations_35_80
# python utils/visualize_preprocess_features.py --input_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData" --output_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\feature_visualizations_35_80" --sample_rate 16000 --num_mel_bins 64 --frame_length 25 --frame_shift 10 --low_freq 3500 --high_freq 8000
#
# 3) 仅处理公开无人机数据（默认参数）
# python utils/visualize_preprocess_features.py --input_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\公开无人机数据" --output_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\feature_visualizations_public_default"
#
# 4) 仅处理公开无人机数据（3500-8000 Hz）
# python utils/visualize_preprocess_features.py --input_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\公开无人机数据" --output_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\feature_visualizations_public_35_80" --sample_rate 16000 --num_mel_bins 64 --frame_length 25 --frame_shift 10 --low_freq 3500 --high_freq 8000
#
# 5) 仅处理自研电麦误告（默认参数）
# python utils/visualize_preprocess_features.py --input_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\自研电麦误告" --output_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\feature_visualizations"
#
# 6) 仅处理自研电麦误告（3500-8000 Hz）
# python utils/visualize_preprocess_features.py --input_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\自研电麦误告" --output_dir "E:\Code\930_Codes\Audio_classify\audioset_tagging_cnn\TestData\feature_visualizations_35_80" --sample_rate 16000 --num_mel_bins 64 --frame_length 25 --frame_shift 10 --low_freq 3500 --high_freq 8000


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTORCH_DIR = PROJECT_ROOT / "pytorch"
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def find_audio_files(input_dir: Path):
    audio_files = []
    skipped_archives = []

    for file_path in input_dir.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            audio_files.append(file_path)
        elif suffix in {".rar", ".zip", ".7z"}:
            skipped_archives.append(file_path)

    return sorted(audio_files), sorted(skipped_archives)


def load_audio(audio_path: Path, target_sr: int):
    waveform, sample_rate = torchaudio.load(str(audio_path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sr,
        )

    return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)


def save_heatmap(feature_map: np.ndarray, save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.imshow(feature_map, aspect="auto", origin="lower", cmap="viridis")
    plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("Feature Bin")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_comparison(preprocess_map: np.ndarray, bn_map: np.ndarray, save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    im0 = axes[0].imshow(preprocess_map, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"{title} - preprocess")
    axes[0].set_ylabel("Feature Bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.02)

    im1 = axes[1].imshow(bn_map, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title(f"{title} - batchnorm")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Feature Bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def preprocess_to_map(preprocess_output: torch.Tensor) -> np.ndarray:
    # preprocess 输出形状: (B, 1, frames, 64)
    return preprocess_output[0, 0].detach().cpu().numpy().T


def batchnorm_to_map(batchnorm_output: torch.Tensor) -> np.ndarray:
    # BN 输入输出形状: (B, 64, frames, 1)
    return batchnorm_output[0, :, :, 0].detach().cpu().numpy()


def custom_preprocess(
    source: torch.Tensor,
    sample_rate: int,
    num_mel_bins: int,
    frame_length: float,
    frame_shift: float,
    low_freq: float,
    high_freq: float,
    fbank_mean: float = 15.41663,
    fbank_std: float = 6.55582,
) -> torch.Tensor:
    waveforms = source.unsqueeze(1) * (2 ** 15)
    fbanks = []

    for i in range(waveforms.size(0)):
        fbank = ta_kaldi.fbank(
            waveforms[i],
            num_mel_bins=num_mel_bins,
            sample_frequency=sample_rate,
            frame_length=frame_length,
            frame_shift=frame_shift,
            low_freq=low_freq,
            high_freq=high_freq,
        )
        fbanks.append(fbank)

    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    fbank = fbank.unsqueeze(1)
    return fbank


def process_audio_file(
    audio_path: Path,
    input_dir: Path,
    output_dir: Path,
    sample_rate: int,
    num_mel_bins: int,
    frame_length: float,
    frame_shift: float,
    low_freq: float,
    high_freq: float,
):
    waveform = load_audio(audio_path, sample_rate)
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)

    with torch.no_grad():
        preprocess_output = custom_preprocess(
            waveform_tensor,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            low_freq=low_freq,
            high_freq=high_freq,
        )
        bn_input = preprocess_output.transpose(1, 3)

        # 使用一个新的 BN 层逐文件归一化，展示该层对通道特征的标准化效果。
        batch_norm = nn.BatchNorm2d(num_mel_bins)
        batch_norm.train()
        batchnorm_output = batch_norm(bn_input)

    preprocess_map = preprocess_to_map(preprocess_output)
    batchnorm_map = batchnorm_to_map(batchnorm_output)

    relative_path = audio_path.relative_to(input_dir)
    stem = relative_path.with_suffix("")

    preprocess_path = output_dir / "preprocess" / f"{stem}.png"
    batchnorm_path = output_dir / "batchnorm" / f"{stem}.png"
    compare_path = output_dir / "compare" / f"{stem}.png"

    save_heatmap(
        preprocess_map,
        preprocess_path,
        f"{audio_path.name} preprocess fbank ({num_mel_bins} bins)",
    )
    save_heatmap(
        batchnorm_map,
        batchnorm_path,
        f"{audio_path.name} after BatchNorm2d({num_mel_bins})",
    )
    save_comparison(
        preprocess_map,
        batchnorm_map,
        compare_path,
        audio_path.name,
    )

    print(f"[OK] {audio_path}")
    print(f"     preprocess: {preprocess_path}")
    print(f"     batchnorm : {batchnorm_path}")
    print(f"     compare   : {compare_path}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="将音频通过 CNNLSTMExtractor.preprocess 和 BatchNorm2d(64) 后保存为可视化图像。"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROJECT_ROOT / "TestData"),
        help="待扫描的输入目录，脚本会递归查找常见音频文件。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "TestData" / "feature_visualizations"),
        help="输出目录，默认保存在 TestData/feature_visualizations 下。",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="音频加载采样率，需与 preprocess 中的 sample_frequency 保持一致。",
    )
    parser.add_argument("--num_mel_bins", type=int, default=64, help="fbank 频带数量。")
    parser.add_argument("--frame_length", type=float, default=25.0, help="帧长，单位 ms。")
    parser.add_argument("--frame_shift", type=float, default=10.0, help="帧移，单位 ms。")
    parser.add_argument("--low_freq", type=float, default=20.0, help="fbank 下限频率，单位 Hz。")
    parser.add_argument("--high_freq", type=float, default=0.0, help="fbank 上限频率，单位 Hz；<=0 表示相对 Nyquist 偏移。")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    audio_files, skipped_archives = find_audio_files(input_dir)

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(
        "fbank 参数: "
        f"num_mel_bins={args.num_mel_bins}, sample_rate={args.sample_rate}, "
        f"frame_length={args.frame_length}, frame_shift={args.frame_shift}, "
        f"low_freq={args.low_freq}, high_freq={args.high_freq}"
    )

    if skipped_archives:
        print("检测到压缩包，已跳过以下文件:")
        for archive_path in skipped_archives:
            print(f"  - {archive_path}")

    if not audio_files:
        print("未找到可处理的音频文件。请先解压压缩包，或把音频放到输入目录后重试。")
        return

    print(f"发现 {len(audio_files)} 个音频文件，开始生成可视化...")

    for audio_path in audio_files:
        try:
            process_audio_file(
                audio_path,
                input_dir,
                output_dir,
                args.sample_rate,
                args.num_mel_bins,
                args.frame_length,
                args.frame_shift,
                args.low_freq,
                args.high_freq,
            )
        except Exception as exc:
            print(f"[FAIL] {audio_path}: {exc}")

    print("处理完成。")


if __name__ == "__main__":
    main()
