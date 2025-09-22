import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import csv

DATASET_ROOT = "dataset_root"
WAV_DIR = os.path.join(DATASET_ROOT, "audios")
META_DIR = os.path.join(DATASET_ROOT, "metadata")
os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

CSV_PATH = os.path.join(META_DIR, "gk_train_segments.csv")

# 如果 CSV 不存在，创建文件
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["audio_name", "start_time", "end_time", "label"])
    df.to_csv(CSV_PATH, index=False)

# 全局变量存放点击的时间点
clicks = []
current_wav = None
ax = None

def annotate_wav(wav_path):
    """绘制波形并交互式标注区间"""
    global clicks, current_wav, ax
    current_wav = wav_path
    clicks = []

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    times = librosa.times_like(y, sr=sr)

    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(os.path.basename(wav_path))

    # 鼠标点击事件
    def onclick(event):
        if event.inaxes != ax:
            return
        clicks.append(event.xdata)
        if len(clicks) == 2:
            start, end = sorted(clicks)
            ax.axvspan(start, end, color="red", alpha=0.3)
            plt.draw()
            print(f"选择区间: {start:.0f} - {end:.0f} 秒，按键 d/n/m 或 Enter 输入标签")

    # 键盘事件
    def onkey(event):
        if len(clicks) == 2:
            start, end = sorted(clicks)
            label = None
            if event.key == "d":
                label = "/m/drone"
            elif event.key == "n":
                label = "/m/noise"
            elif event.key == "m":
                label = "/m/missile"
            elif event.key == "enter":
                label = input(f"请输入标签 (区间 {start:.0f}-{end:.0f}s): ")

            if label:
                # 将current_wav路径改为最后一层文件夹+文件名
                current_wav_ = os.path.basename(os.path.dirname(current_wav)) + "/" + os.path.basename(current_wav)
                save_annotation(current_wav_, start, end, label)
                clicks.clear()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

def save_annotation_err(wav_path, start, end, label):
    global ax
    """保存标注到 CSV"""
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([{
        "audio_name": wav_path,
        "start_time": int(start),
        "end_time": int(end),
        "label": f"{label}"
    }])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ 已保存标注: {wav_path}, {int(start)}-{int(end)}s, {label}")
    # 在波形上标注区间和文字
    if ax is not None:
        ax.axvspan(start, end, color="green", alpha=0.3)
        mid = (start + end) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.8, label,
                color="black", fontsize=10, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        plt.draw()

def save_annotation(wav_path, start, end, label):
    """
    将单条标注追加写入 CSV（PANNs 风格）并在当前图上绘制标注。
    输出行格式示例:
    Missile/20250909_2_Channel_0.wav, 24, 29, "/m/missile"
    """
    global ax, CSV_PATH, DATASET_ROOT

    # --- 1) 规范化 audio_name：尝试转成相对路径并使用正斜杠 ---
    try:
        # 如果 DATASET_ROOT 未定义或 wav_path 已经是相对路径，这里会抛异常或返回相同
        audio_rel = wav_path.replace('.wav', '')
    except Exception:
        audio_rel = wav_path
    audio_rel = os.path.normpath(audio_rel).replace("\\", "/")

    # --- 2) 时间格式化：整数优先，非整数保留 3 位小数 ---
    def _fmt_time(t):
        try:
            tf = int(t)
        except Exception:
            tf = 0.0
        if abs(tf - round(tf)) < 1e-9:
            return str(int(round(tf)))
        else:
            return f"{tf:.3f}"

    start_s = _fmt_time(start)
    end_s = _fmt_time(end)

    # --- 3) 处理 label：CSV 规范中双引号需要被内嵌双引号转义 ---
    label_str = str(label)
    label_escaped = label_str.replace('"', '""')  # CSV 内部双引号转义为两个双引号

    # --- 4) 确保 CSV 存在且有正确表头；否则写表头 ---
    header_line = "audio_name, start_time, end_time, label\n"
    need_write_header = False
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        need_write_header = True
    else:
        try:
            with open(CSV_PATH, "r", encoding="utf-8") as fh:
                first = fh.readline()
            if not first.lower().lstrip().startswith("audio_name"):
                need_write_header = True
        except Exception:
            need_write_header = True

    # --- 5) 写入（追加或新建）——化为期望的单行格式（逗号后带空格，label 带双引号） ---
    line = f'{audio_rel}, {start_s}, {end_s}, "{label_escaped}"\n'
    mode = "a" if not need_write_header else "w"
    with open(CSV_PATH, mode, encoding="utf-8", newline="\n") as fh:
        if need_write_header:
            fh.write(header_line)
        fh.write(line)

    print(f'✅ 已保存标注: {audio_rel}, {start_s}-{end_s}s, {label_str}')

    # --- 6) 在当前图上绘制标注（保留原来的显示行为） ---
    if 'ax' in globals() and ax is not None:
        try:
            ax.axvspan(start, end, color="green", alpha=0.3)
            mid = (start + end) / 2
            ytop = ax.get_ylim()[1]
            ax.text(mid, ytop * 0.8, label_str,
                    color="black", fontsize=10, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            plt.draw()
        except Exception as e:
            # 不让显示错误中断主流程
            print("⚠️ 绘图标注失败:", e)

def main():
    wav_files = []
    for root, _, files in os.walk(WAV_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"发现 {len(wav_files)} 个 wav 文件，逐个标注...")

    for wav_path in wav_files:
        annotate_wav(wav_path)


if __name__ == "__main__":
    main()
