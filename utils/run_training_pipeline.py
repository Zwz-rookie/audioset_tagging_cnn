#!/usr/bin/env python3
"""
训练流程脚本，用于自动执行完整的训练和测试流程
包括：
1. 生成HDF5文件
2. 生成索引文件
3. 模型训练
4. 模型测试
"""

import os
import subprocess
import sys
os.environ['AUDIO_CLASSIFY_DATA_MODE'] = 'SEA'

def run_command(command, description):
    """
    执行命令并输出结果
    """
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {' '.join(command)}")
    print('='*50)
    
    try:
        # 执行命令
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
        )
        
        print("\n执行成功:")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n执行失败: {e}")
        print("错误输出:")
        print(e.stderr)
        return False
        
    except Exception as e:
        print(f"\n执行失败: {e}")
        return False

def main():
    """
    主函数，执行完整训练流程
    """
    print("开始执行完整训练流程")
    
    # 1. 生成HDF5文件
    generate_hdf5_cmd = [
        "python", "utils/dataset.py", "pack_waveforms_to_hdf5",
        "--csv_path", "dataset_root/metadata/gk_train_segments.csv",
        "--audios_dir", "dataset_root/audios/balanced_train_segments",
        "--waveforms_hdf5_path", "dataset_hdf5/hdf5s/waveforms/balanced_train.h5"
    ]
    
    if not run_command(generate_hdf5_cmd, "生成HDF5文件"):
        print("HDF5文件生成失败，停止执行")
        sys.exit(1)
    
    # 2. 生成索引文件
    generate_indexes_cmd = [
        "python", "utils/create_indexes.py", "create_indexes",
        "--waveforms_hdf5_path", "dataset_hdf5/hdf5s/waveforms/balanced_train.h5",
        "--indexes_hdf5_path", "dataset_hdf5/hdf5s/indexes/balanced_train.h5"
    ]
    
    if not run_command(generate_indexes_cmd, "生成索引文件"):
        print("索引文件生成失败，停止执行")
        sys.exit(1)
    
    # 3. 模型训练
    train_cmd = [
        "python", "pytorch/train_epoch.py", "train",
        "--data_type", "balanced_train",
        "--checkpoint_path", "MobileNetV2_Mod.pth",
        "--workspace", "dataset_hdf5",
        "--sample_rate", "8000",
        "--window_size", "1024",
        "--hop_size", "320",
        "--mel_bins", "64",
        "--fmin", "50",
        "--fmax", "14000",
        "--model_type", "MobileNetV2_Mod",
        "--loss_type", "clip_bce",
        "--balanced", "balanced",
        "--augmentation", "none",
        "--batch_size", "32",
        "--learning_rate", "1e-3",
        "--resume_iteration", "0",
        "--early_stop", "1000000",
        "--cuda"
    ]
    
    if not run_command(train_cmd, "模型训练"):
        print("模型训练失败，停止执行")
        sys.exit(1)
    
    # 4. 模型测试
    test_cmd = [
        "python", "pytorch/inference.py", "audio_tagging",
        "--sample_rate", "8000",
        "--model_type", "MobileNetV2_Mod",
        "--checkpoint_path", "MobileNetV2_Mod.pth",
        "--audio_path", "resources/Channel_2_Pos_1009.wav",
        "--cuda"
    ]
    
    if not run_command(test_cmd, "模型测试"):
        print("模型测试失败，停止执行")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✅ 完整训练流程执行成功！")
    print("="*50)

if __name__ == "__main__":
    main()
