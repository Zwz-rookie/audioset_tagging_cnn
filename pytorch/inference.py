import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config
import time
import torchaudio

def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    if not model_type.endswith("_Mod"):
        model = Model(sample_rate=sample_rate, window_size=window_size,
                      hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                      classes_num=classes_num)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print("✅ 成功加载 Cnn14_16k 模型！")

    elif model_type == 'MobileNetV2_Mod':
        if "Mod" in checkpoint_path:
            # model = torch.load("MobileNetV2_Mod_trace.pt", map_location=device)
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print("✅ 成功加载 MobileNetV2_Mod 模型！")
        else:
            # 2. 初始化新模型
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            model_state = model.state_dict()
            # 1. 加载预训练模型权重
            pretrained = torch.load(checkpoint_path, map_location=device)
            pretrained_state = pretrained["model"] if "model" in pretrained else pretrained
            # 3. 过滤掉前端特征提取层，只保留 bn0 及之后的
            filtered_state = {
                k: v for k, v in pretrained_state.items()
                if not (k.startswith("feature_extractor") or k.startswith("fc_audioset")
                        or k.startswith("spectrogram_extractor") or k.startswith("logmel_extractor"))
            }
            # 4. 更新 state_dict
            model_state.update(filtered_state)
            # 5. 加载
            model.load_state_dict(model_state)
            print("✅ 成功加载 MobileNetV2 bn0 及后续层参数！")
    else:
        if "Mod" in checkpoint_path:
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print("✅ 成功加载 Cnn14_16k_Mod 模型！")
        else:
            # 1. 加载预训练模型权重
            pretrained = torch.load(checkpoint_path, map_location=device)
            pretrained_state = pretrained["model"] if "model" in pretrained else pretrained

            # 2. 初始化新模型
            model = Model(mel_bins=mel_bins, classes_num=classes_num)
            model_state = model.state_dict()

            # 3. 过滤掉前端特征提取层，只保留 bn0 及之后的
            filtered_state = {
                k: v for k, v in pretrained_state.items()
                if k.startswith("bn0") or k.startswith("conv_block") or k.startswith("fc")
            }

            # 4. 更新 state_dict
            model_state.update(filtered_state)

            # 5. 加载
            model.load_state_dict(model_state)

            print("✅ 成功加载 bn0 及后续层参数！")

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        # model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, 0:32000]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        # example_input = waveform
        # traced = torch.jit.trace(model, example_input, strict=False)
        # traced.save("MobileNetV2_Mod_trace.pt")
        print('Inference start...')
        start_time = time.time()
        batch_output_dict = model(waveform)
        print('Inference time: {:.3f} seconds'.format(time.time() - start_time))

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(3):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels

def pcm_to_resampled_waveform(data, src_sample_rate=4000, target_sample_rate=16000):
    """
    将原始 PCM 数据（4kHz）直接转换为 16kHz waveform (numpy array)，避免存储到 wav 再读取。

    参数:
    - data: numpy array[int16] 或 bytes (PCM 波形数据)
    - src_sample_rate: PCM 原始采样率 (默认 4000Hz)
    - target_sample_rate: 目标采样率 (默认 16000Hz)

    返回:
    - waveform: numpy array[float32], shape=(n_samples,)
    """

    # 转换成 float32 [-1, 1]
    waveform = data #.astype(np.float32) / 32768.0

    # 重采样（4k → 16k）
    if src_sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=src_sample_rate,
            target_sr=target_sample_rate,
            res_type="kaiser_best"  # 更快，可以换 "kaiser_best" 提高质量
        )

    return waveform


def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    start_time = time.time()
    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
    print('Inference time: {:.3f} seconds'.format(time.time() - start_time))

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 10  # Show top results
    top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
        hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=16000)
    parser_at.add_argument('--window_size', type=int, default=512)
    parser_at.add_argument('--hop_size', type=int, default=160)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=8000)
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')