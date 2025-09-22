import os
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 输入输出目录
DATASET_ROOT = "dataset_root"
MP3_DIR = os.path.join(DATASET_ROOT, "audios_mp3")
WAV_DIR = os.path.join(DATASET_ROOT, "audios")
META_DIR = os.path.join(DATASET_ROOT, "metadata")
os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

CSV_PATH = os.path.join(META_DIR, "gk_train_segments.csv")

# CSV 文件初始化
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["audio_name", "label"])
    df.to_csv(CSV_PATH, index=False)

def mp3_to_wav_and_plot(mp3_path, wav_path, sr=16000):
    """将 mp3 转为 wav，并画出波形图"""
    # 读取 mp3
    y, _ = librosa.load(mp3_path, sr=sr, mono=True)

    # 保存 wav
    sf.write(wav_path, y, sr)

    # 可视化
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(os.path.basename(wav_path))
    plt.tight_layout()

    # 保存图像，方便标注时查看
    png_path = wav_path.replace(".wav", ".png")
    plt.savefig(png_path)
    plt.show()
    plt.close()

def mp3_to_wav_and_plot2(mp3_path, wav_path, sr=16000):
    """
    将 mp3 转换成 wav (16kHz)，并绘制波形
    X轴：波形采样点索引（0, 1, 2, ...）
    Y轴：振幅
    """
    # 读取 mp3
    y, _ = librosa.load(mp3_path, sr=sr, mono=True)

    # 保存 wav
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    sf.write(wav_path, y, sr)

    # X 轴为索引
    x = np.arange(len(y))

    # 绘制波形
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, linewidth=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title(os.path.basename(wav_path))
    plt.tight_layout()

    # 保存图片
    png_path = wav_path.replace(".wav", ".png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"处理完成: {mp3_path} -> {wav_path}, 可视化保存为 {png_path}")
def main():
    mp3_files = []
    for root, _, files in os.walk(MP3_DIR):
        for f in files:
            if f.lower().endswith(".mp3"):
                mp3_files.append(os.path.join(root, f))

    print(f"发现 {len(mp3_files)} 个 mp3 文件，开始处理...")

    for mp3_path in mp3_files:
        rel_path = os.path.relpath(mp3_path, MP3_DIR)  # 相对路径
        wav_path = os.path.join(WAV_DIR, os.path.splitext(rel_path)[0] + ".wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        mp3_to_wav_and_plot2(mp3_path, wav_path, sr=16000)

    print("全部 mp3 转换完成，wav 和波形图已生成。")
    print(f"标注 CSV 文件在: {CSV_PATH}")

if __name__ == "__main__":
    main()
