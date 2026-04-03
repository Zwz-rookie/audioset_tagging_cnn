# AGENTS.md

本文件用于指导后续在本工程内的开发、训练与推理工作。

## 项目概览
- 本工程基于 PANNs，用于音频标签与声音事件检测的训练、推理与模型导出。
- 主要入口位于 `pytorch/`（训练、推理、模型定义）与 `utils/`（数据处理、索引、工具脚本）。

## 目录结构速览（按当前工程目录）
- `pytorch/`：模型、训练、评估、推理与 PyTorch 相关工具。
- `utils/`：数据处理、索引、训练流水线、统计绘图与通用工具。
- `dataset_root/`、`dataset_root_GM/`、`dataset_root_DM/`：原始音频与 metadata（CSV）。
- `dataset_hdf5/`、`dataset_hdf5_GM/`：HDF5 波形与索引（训练输入）。
- `scripts/`：原作者提供的下载/打包/训练 shell 脚本。
- `resources/`：示例音频、图片等资源。
- `metadata/`：与数据/统计相关的元信息。
- `_logs/`：下载或训练相关日志输出。
- `fbank_visuals/`：可视化输出。
- `*.pth`、`*.pt`：预训练或已导出的模型权重与 Trace 模型。

## utils 目录脚本作用
- `utils/ai_flywheel.py`：GM 模式的“AI 飞轮”脚本，监听新增音频数量并触发训练（通过 `config.py` 读取路径与阈值）。
- `utils/ai_flywheel_DM.py`：DM 模式“AI 飞轮”版本，逻辑与 GM 类似，切换数据模式。
- `utils/ai_flywheel_sea.py`：SEA 模式“AI 飞轮”版本。
- `utils/config.py`：全局配置与标签表加载。根据环境变量 `AUDIO_CLASSIFY_DATA_MODE` 选择数据路径与标签表，并构建 `labels` / `ids` / `classes_num` 等常量。
- `utils/crash.py`：IPython 异常处理钩子，用于崩溃时进入调试输出。
- `utils/create_black_list.py`：生成黑名单 CSV（如 DCASE2017 task4），供训练时跳过特定样本。
- `utils/create_indexes.py`：从 HDF5 波形文件创建索引；也可合并为 full_train 的索引。
- `utils/data_generator.py`：数据集与采样器（TrainSampler、BalancedTrainSampler 等），从索引 HDF5 读取波形/标签。
- `utils/data_generator_my.py`：`data_generator.py` 的定制版本，供 `pytorch/train_epoch.py` 使用。
- `utils/dataset.py`：数据集处理主脚本，含下载音频、拆分 CSV、打包波形到 HDF5 等。
- `utils/delete_wavs_by_csv_range.py`：根据 CSV 行号范围删除或定位 wav 文件的辅助脚本。
- `utils/plot_for_paper.py`：生成论文用统计图（mAP 等）。
- `utils/plot_statistics.py`：读取统计结果并绘制曲线或导出指标。
- `utils/run_training_pipeline.py`：一键训练流水线（生成 HDF5 -> 索引 -> 训练 -> 测试）。
- `utils/utilities.py`：通用工具函数（日志、读元数据、混合采样、统计保存等）。

## pytorch 目录脚本作用
- `pytorch/check_device_err.py`：检查自实现 rfft 与 PyTorch/CPU-GPU 结果一致性。
- `pytorch/evaluate.py`：评估器，调用 `pytorch_utils.forward` 计算 mAP/AUC。
- `pytorch/finetune_template.py`：迁移学习模板（基于 Cnn14 作为底座）。
- `pytorch/gk_kaldi.py`：自实现的 kaldi 特征相关函数（fbank/mfcc/梅尔等）。
- `pytorch/inference.py`：推理入口，支持 audio tagging / sound event detection；包含多模型权重加载逻辑。
- `pytorch/losses.py`：损失函数封装（当前主要为 `clip_bce`）。
- `pytorch/main.py`：主训练入口（PANNs 训练流程）。
- `pytorch/models.py`：模型结构定义（Cnn、ResNet、MobileNet、Wavegram 等）。
- `pytorch/pytorch_utils.py`：训练与推理通用工具（move_data_to_device、mixup、forward、插值等）。
- `pytorch/train_epoch.py`：自定义训练逻辑版本，支持模型序列化（trace 导出 CPU/GPU 版本）。

## 常用入口与流程
- 训练：`pytorch/main.py`（标准训练流程）或 `pytorch/train_epoch.py`（定制训练/导出）。
- 推理：`pytorch/inference.py`。
- 数据准备：`utils/dataset.py`（下载与打包）与 `utils/create_indexes.py`（索引）。
- 一键流程：`utils/run_training_pipeline.py`。

## 环境与配置注意事项
- `utils/config.py` 内含 Windows 路径（如 `E:\Code\...`），在非 Windows 环境运行前需要调整。
- 数据模式由 `AUDIO_CLASSIFY_DATA_MODE` 控制（如 GM / SEA / DM）。
- `AUDIO_CLASSIFY_TEST`、`AUDIO_CLASSIFY_VAL` 在 `pytorch/train_epoch.py` 中用于控制测试/验证逻辑。

## 产物与输出
- 训练权重：`*.pth`，通常由训练脚本输出。
- Trace 模型：`*_trace.pt` / `*_trace_cuda.pt`，由 `pytorch/train_epoch.py` 中序列化逻辑生成。
- 日志与统计：`_logs/` 与 `statistics/` 下生成。
