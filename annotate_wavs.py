import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import pandas as pd
import simpleaudio as sa
import sounddevice as sd

DATASET_ROOT = "dataset_root_GM"
WAV_DIR = os.path.join(DATASET_ROOT, "audios")
META_DIR = os.path.join(DATASET_ROOT, "metadata")
os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

CSV_PATH = os.path.join(META_DIR, "gk_train_segments_GM.csv")

if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["audio_name", "start_time", "end_time", "label"])
    df.to_csv(CSV_PATH, index=False)

clicks = []
current_wav = None
ax = None
current_play = None   # 保存当前播放对象


def play_audio(y, sr):
    """播放音频"""
    global current_play
    if current_play:
        current_play.stop()

    # 重采样到44100Hz，这是最常见的采样率之一
    target_sr = 44100
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio_data = (y * 32767).astype("int16")  # 转为16位PCM
    current_play = sa.play_buffer(audio_data, 1, 2, sr)


# def play_segment(start, end):
#     """播放音频片段"""
#     global y_global, sr_global
#     start_idx = int(start * sr_global)
#     end_idx = int(end * sr_global)
#     sd.stop()
#     sd.play(y_global[start_idx:end_idx], sr_global)
#
# def add_play_button(fig, start, end, ypos=1.05):
#     """在波形图上方添加播放按钮"""
#     ax_button = plt.axes([0.1, ypos, 0.1, 0.05])  # [left, bottom, width, height]
#     btn = Button(ax_button, f"▶ {int(start)}-{int(end)}s")
#     btn.on_clicked(lambda event: play_segment(start, end))
#     return btn


def annotate_wav(wav_path):
    """绘制波形并交互式标注区间 + 播放器控件"""
    global clicks, current_wav, ax, current_play
    current_wav = wav_path
    clicks = []

    # --- 加载音频 ---
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr  # 音频时长

    # --- 画波形 ---
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(os.path.basename(wav_path))

    # --- 功能 1：加载已保存标注 ---
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        full_pth = os.path.dirname(wav_path)
        # 如果"_GM"在路径中，则保留倒数两级目录+文件
        if '_GM' in full_pth:
            rel_name = os.path.basename(os.path.dirname(full_pth)) + "/" + os.path.basename(full_pth) + "/" + os.path.basename(wav_path)
        else:
            rel_name = os.path.basename(full_pth) + "/" + os.path.basename(wav_path)
        rel_name = rel_name.replace('.wav', '')
        prev_annots = df[df["audio_name"] == rel_name]
        for _, row in prev_annots.iterrows():
            start, end, label = row["start_time"], row["end_time"], row["label"]
            ax.axvspan(start, end, color="green", alpha=0.3)
            mid = (start + end) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.8, label,
                    color="black", fontsize=10, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        if len(prev_annots) > 0:
            print(f"📂 已加载 {len(prev_annots)} 条历史标注")

    # --- 鼠标点击标注 ---
    def onclick(event):
        if event.inaxes != ax:
            return
        clicks.append(event.xdata)
        if len(clicks) == 2:
            start, end = sorted(clicks)
            ax.axvspan(start, end, color="red", alpha=0.3)
            plt.draw()
            print(f"选择区间: {start:.0f} - {end:.0f} 秒，按键 d/n/m 或 Enter 输入标签")

    # --- 键盘输入标签 ---
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
            elif event.key == "w":
                label = "/m/quadrotor"
            elif event.key == "1":
                label = "/m/1Khz40db"
            elif event.key == "enter":
                label = input(f"请输入标签 (区间 {start:.0f}-{end:.0f}s): ")

            if label:
                # 将current_wav路径改为最后一层文件夹+文件名
                full_pth = os.path.dirname(wav_path)
                if '_GM' in full_pth:
                    current_wav_ = os.path.basename(os.path.dirname(full_pth)) + "/" + os.path.basename(full_pth) + "/" + os.path.basename(wav_path)
                else:
                    current_wav_ = os.path.basename(full_pth) + "/" + os.path.basename(wav_path)
                save_annotation(current_wav_, start, end, label)
                clicks.clear()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()


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
        print(f"📂 正在标注 {wav_path}")
        annotate_wav(wav_path)


if __name__ == "__main__":
    main()
