import os
# 设置环境变量为GM，确保使用GM配置
os.environ['AUDIO_CLASSIFY_DATA_MODE'] = 'GM'

import time
import csv
import glob
import hashlib
import shutil
from datetime import datetime
import subprocess
import config


class AIFlywheel:
    def __init__(self):
        # 配置路径
        self.audio_dir = config.audio_dir
        self.metadata_dir = config.metadata_dir
        self.class_labels_file = config.class_labels_file
        self.train_segments_file = config.train_segments_file
        self.state_file = config.state_file
        self.output_model_pth = config.output_model_pth
        
        # 阈值配置
        self.increment_threshold = 150
        
        # 训练状态跟踪
        self.is_training = False
        
        # 加载类别标签
        self.class_labels = self._load_class_labels()
        
        # 加载状态
        self.previous_count = self._load_state()
        
        print("AI飞轮系统初始化完成")
        print(f"音频目录: {self.audio_dir}")
        print(f"类别标签文件: {self.class_labels_file}")
        print(f"训练片段文件: {self.train_segments_file}")
        print(f"增量阈值: {self.increment_threshold} 个文件")

    def _load_class_labels(self):
        """加载类别标签"""
        class_labels = {}
        try:
            with open(self.class_labels_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    display_name = row['display_name']
                    mid = row['mid']
                    class_labels[display_name.lower()] = mid
            print(f"成功加载 {len(class_labels)} 个类别标签")
        except Exception as e:
            print(f"加载类别标签失败: {e}")
        return class_labels

    def _load_state(self):
        """加载上一次扫描的状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return int(f.read().strip())
            except Exception as e:
                print(f"加载状态文件失败: {e}")
        return 0

    def _save_state(self, count):
        """保存当前状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                f.write(str(count))
        except Exception as e:
            print(f"保存状态文件失败: {e}")

    def _count_wav_files(self):
        """统计wav文件数量"""
        try:
            wav_files = glob.glob(os.path.join(self.audio_dir, "**", "*.wav"), recursive=True)
            return len(wav_files)
        except Exception as e:
            print(f"统计wav文件数量失败: {e}")
            return 0

    def _get_all_wav_files(self):
        """获取所有wav文件路径"""
        try:
            return glob.glob(os.path.join(self.audio_dir, "**", "*.wav"), recursive=True)
        except Exception as e:
            print(f"获取wav文件列表失败: {e}")
            return []

    def _get_existing_records(self):
        """获取已存在的记录"""
        existing_records = set()
        if os.path.exists(self.train_segments_file):
            try:
                with open(self.train_segments_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_records.add(row['audio_name'])
            except Exception as e:
                print(f"加载现有记录失败: {e}")
        return existing_records

    def _add_new_records(self, new_wav_files):
        """添加新记录到训练片段文件"""
        existing_records = self._get_existing_records()
        new_records = []

        for file_path in new_wav_files:
            # 获取相对路径（相对于audio_dir）
            relative_path = os.path.relpath(file_path, self.audio_dir)
            # 将反斜杠转换为斜杠
            relative_path = relative_path.replace('\\', '/')
            
            # 检查是否已存在（需要考虑是否包含.wav后缀）
            relative_path_no_ext = os.path.splitext(relative_path)[0]
            if relative_path in existing_records or relative_path_no_ext in existing_records:
                continue
            
            # 提取类别
            parts = relative_path.split('/')
            if len(parts) < 2:
                continue
                
            class_name = parts[0].lower()
            if class_name not in self.class_labels:
                print(f"警告: 未知类别 '{class_name}'，跳过文件 '{relative_path}'")
                continue
                
            mid = self.class_labels[class_name]
            
            # 创建新记录（移除.wav后缀）
            audio_name = relative_path_no_ext
            record = {
                'audio_name': audio_name,
                'start_time': '0',
                'end_time': '4',
                'label': mid
            }
            
            new_records.append(record)

        if new_records:
            print(f"发现 {len(new_records)} 个新文件，正在添加到标注文件...")
            
            # 检查文件是否存在
            file_exists = os.path.exists(self.train_segments_file)
            
            # 检查文件末尾是否需要添加换行符
            need_newline = False
            if file_exists and os.path.getsize(self.train_segments_file) > 0:
                with open(self.train_segments_file, 'rb') as f:
                    f.seek(-1, 2)
                    last_byte = f.read(1)
                    need_newline = last_byte != b'\n'
            
            with open(self.train_segments_file, 'a', encoding='utf-8') as f:
                # 如果是第一次写入数据（文件不存在或为空），先检查是否需要表头
                if not file_exists or os.path.getsize(self.train_segments_file) == 0:
                    # 写入表头
                    f.write('audio_name,start_time,end_time,label\n')
                elif need_newline:
                    # 对于已存在文件，在添加第一行数据前添加回车
                    f.write('\n')
                
                # 写入记录，使用指定格式
                for record in new_records:
                    audio_rel = record['audio_name']
                    start_s = record['start_time']
                    end_s = record['end_time']
                    label_escaped = record['label']
                    line = f'{audio_rel}, {start_s}, {end_s}, "{label_escaped}"\n'
                    f.write(line)
            
            print(f"成功添加 {len(new_records)} 条新标注记录")
        else:
            print("没有发现需要添加的新记录")

    def _generate_hdf5(self):
        """生成HDF5文件"""
        print("开始生成HDF5文件...")
        command = [
            "python", "utils/dataset.py", "pack_waveforms_to_hdf5",
            "--csv_path", "dataset_root_GM/metadata/gk_train_segments_GM.csv",
            "--audios_dir", "dataset_root_GM/audios/balanced_train_segments_GM",
            "--waveforms_hdf5_path", "dataset_hdf5_GM/hdf5s/waveforms/balanced_train.h5"
        ]
        
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # subprocess.STDOUT合并标准输出和错误输出
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
            )
            print("HDF5文件生成成功")
            print("输出信息:")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"HDF5文件生成失败: {e}")
            print("错误信息:")
            print(e.stderr)
            return False
        except Exception as e:
            print(f"HDF5文件生成失败: {e}")
            return False
    
    def _generate_indexes(self):
        """生成索引文件"""
        print("开始生成索引文件...")
        command = [
            "python", "utils/create_indexes.py", "create_indexes",
            "--waveforms_hdf5_path", "dataset_hdf5_GM/hdf5s/waveforms/balanced_train.h5",
            "--indexes_hdf5_path", "dataset_hdf5_GM/hdf5s/indexes/balanced_train.h5"
        ]
        
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # subprocess.STDOUT合并标准输出和错误输出
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
            )
            print("索引文件生成成功")
            print("输出信息:")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"索引文件生成失败: {e}")
            print("错误信息:")
            print(e.stderr)
            return False
        except Exception as e:
            print(f"索引文件生成失败: {e}")
            return False
    
    def _trigger_model_training(self, checkpoint_path=None):
        """触发模型训练"""
        print("\n开始模型训练...")
        command = [
            "python", "pytorch/train_epoch.py", "train",
            "--data_type", "balanced_train",
            "--workspace", "dataset_hdf5_GM",
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
            "--batch_size", "128",
            "--learning_rate", "1e-3",
            "--resume_iteration", "0",
            "--early_stop", "100000",
            "--patience", "20",
            "--cuda"
        ]
        
        # 添加checkpoint路径参数（如果提供）
        if checkpoint_path:
            command.extend(["--checkpoint_path", checkpoint_path])
        
        # 设置训练状态为正在进行
        self.is_training = True
        
        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
            )
            print("模型训练成功")
            print("输出信息:")
            print(result.stdout)
            self.is_training = False  # 训练完成，重置训练状态

            # 训练完成后，将序列化的模型文件复制到指定目录
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            target_dir = self.output_model_pth

            # 获取源文件路径
            source_checkpoint_name = "MobileNetV2_Mod_GM"
            cpu_model_file = os.path.join(project_dir, f"{source_checkpoint_name}_trace.pt")
            gpu_model_file = os.path.join(project_dir, f"{source_checkpoint_name}_trace_cuda.pt")
            model_file = os.path.join(project_dir, f"{source_checkpoint_name}.pth")

            # 确保目标目录存在
            os.makedirs(target_dir, exist_ok=True)

            # 复制模型文件到目标目录
            try:
                if os.path.exists(cpu_model_file):
                    shutil.copy2(cpu_model_file, target_dir)
                    print(f"✅ CPU模型文件已复制到: {target_dir}")

                if os.path.exists(gpu_model_file):
                    shutil.copy2(gpu_model_file, target_dir)
                    print(f"✅ GPU模型文件已复制到: {target_dir}")

                # if os.path.exists(model_file):
                #     shutil.copy2(model_file, target_dir)
                #     print(f"✅ 模型文件已复制到: {target_dir}")

            except Exception as e:
                print(f"❌ 复制模型文件: {e}")

            return True
        except subprocess.CalledProcessError as e:
            print(f"模型训练失败: {e}")
            print("输出信息:")
            print(e.output)
            self.is_training = False  # 训练失败，重置训练状态
            return False
        except UnicodeDecodeError as e:
            print(f"模型训练过程中编码错误: {e}")
            print("尝试使用GBK编码解码输出:")
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # 合并标准输出和错误输出
                    text=True,
                    encoding='gbk',
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
                )
                print(result.stdout)
                self.is_training = False  # 训练完成，重置训练状态
                return True
            except Exception as e2:
                print(f"GBK编码解码也失败: {e2}")
                self.is_training = False  # 训练失败，重置训练状态
                return False
        except Exception as e:
            print(f"模型训练失败: {e}")
            self.is_training = False  # 训练失败，重置训练状态
            return False
    
    def _trigger_data_compression(self):
        """触发数据集压缩脚本"""
        print("\n开始数据集压缩处理...")
        
        # 生成HDF5文件
        if not self._generate_hdf5():
            return False
        
        # 生成索引文件
        if not self._generate_indexes():
            return False
        
        print("\n数据集压缩处理完成")
        return True
    
    def scan_and_process(self):
        """扫描并处理wav文件增量"""
        print(f"\n开始扫描 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        # 检查是否正在训练，如果是则跳过本次扫描
        if self.is_training:
            print("模型正在训练中，跳过本次扫描")
            return False
            
        # 统计当前wav文件数量
        current_count = self._count_wav_files()
        print(f"当前wav文件数量: {current_count}")
        print(f"上一次扫描数量: {self.previous_count}")
        
        # 检查增量
        increment = current_count - self.previous_count
        print(f"文件增量: {increment}")
        
        if increment >= self.increment_threshold:
            print(f"文件增量达到 {increment}，超过阈值 {self.increment_threshold}，开始处理...")
            
            # 获取所有wav文件
            all_wav_files = self._get_all_wav_files()
            
            # 添加新记录
            self._add_new_records(all_wav_files)
            
            # 触发数据集压缩
            if self._trigger_data_compression():
                # 数据集压缩成功后，触发模型训练
                self._trigger_model_training("MobileNetV2_Mod_GM.pth")
                # self._trigger_model_training()
            
            # 更新状态
            self.previous_count = current_count
            self._save_state(current_count)
            
            print("处理完成")
        else:
            print(f"文件增量 {increment} 未达到阈值 {self.increment_threshold}，不进行处理")

    def run_continuous(self, interval=3600):
        """持续运行，定时扫描"""
        print(f"开始持续监控，扫描间隔: {interval} 秒")
        try:
            while True:
                self.scan_and_process()
                print(f"等待 {interval} 秒后再次扫描...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nAI飞轮系统已停止")


if __name__ == "__main__":
    # 示例用法
    flywheel = AIFlywheel()
    
    # 立即执行一次扫描
    flywheel.scan_and_process()
    
    # 取消注释以下代码以持续运行（定时扫描）
    flywheel.run_continuous(interval=3600)  # 每小时扫描一次