import argparse
import os
import sys
from pathlib import Path

import torch


def _setup_sys_path():
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "pytorch"))
    sys.path.insert(0, str(project_root / "utils"))
    return project_root


def _parse_args(default_data_mode):
    parser = argparse.ArgumentParser(description="Serialize a trained model to TorchScript.")
    parser.add_argument(
        "--data_mode",
        type=str,
        default=default_data_mode,
        choices=["SEA", "GM", "DM"],
        help="Dataset mode to load config variables.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to best_model.pth (optional; defaults to mode-based hard-coded paths).",
    )
    parser.add_argument(
        "--source_checkpoint_path",
        type=str,
        default=None,
        help="Original checkpoint path used for naming outputs (defaults to checkpoint_path).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model class name defined in pytorch/models.py.",
    )
    parser.add_argument("--mel_bins", type=int, default=64)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for .pt files and copied .pth. Defaults to a safe path.",
    )
    parser.add_argument(
        "--example_input_len",
        type=int,
        default=32000,
        help="Example audio length for tracing.",
    )
    return parser.parse_args()


def _resolve_output_dir(project_root, config, output_dir_arg):
    if output_dir_arg:
        return Path(output_dir_arg).resolve()

    candidate = None
    output_model_pth = getattr(config, "output_model_pth", None)
    if output_model_pth and os.path.isdir(output_model_pth):
        candidate = Path(output_model_pth).resolve()

    return candidate if candidate else project_root


def serialize_model(checkpoint_path, model_type, mel_bins, classes_num, output_dir, source_checkpoint_path, example_input_len):
    print("开始模型序列化...")
    try:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print("加载模型到CPU...")
        import importlib
        models = importlib.import_module("models")
        Model = getattr(models, model_type, None)
        if Model is None:
            raise ValueError(f"Unknown model_type: {model_type}")
        if not model_type.endswith("_Mod"):
            model = Model(
                sample_rate=8000,
                window_size=1024,
                hop_size=320,
                mel_bins=mel_bins,
                fmin=50,
                fmax=14000,
                classes_num=classes_num,
            )
        else:
            model = Model(mel_bins=mel_bins, classes_num=classes_num)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()

        print("创建示例输入...")
        example_input_single = torch.randn(1, example_input_len)
        example_input_b10 = torch.randn(10, example_input_len)

        source_checkpoint_filename = os.path.basename(source_checkpoint_path)
        source_checkpoint_name = os.path.splitext(source_checkpoint_filename)[0]

        print("序列化CPU版本模型...")
        cpu_output_path = output_dir / f"{source_checkpoint_name}_trace.pt"
        cpu_output_path_b10 = output_dir / f"{source_checkpoint_name}_trace_b10.pt"
        traced_cpu = torch.jit.trace(model, example_input_single, strict=False)
        traced_cpu.save(str(cpu_output_path))
        print(f"✅ CPU版本模型已保存到: {cpu_output_path}")
        traced_cpu_b10 = torch.jit.trace(model, example_input_b10, strict=False)
        traced_cpu_b10.save(str(cpu_output_path_b10))
        print(f"✅ CPU版本模型(B=10)已保存到: {cpu_output_path_b10}")

        if torch.cuda.is_available():
            print("序列化GPU版本模型...")
            try:
                gpu_output_path = output_dir / f"{source_checkpoint_name}_trace_cuda.pt"
                model.cuda()
                example_input_cuda = example_input_single.cuda()
                example_input_cuda_b10 = example_input_b10.cuda()
                traced_cuda = torch.jit.trace(model, example_input_cuda, strict=False)
                try:
                    traced_cuda.save(str(gpu_output_path))
                    print(f"✅ GPU版本模型已保存到: {gpu_output_path}")
                    gpu_output_path_b10 = output_dir / f"{source_checkpoint_name}_trace_b10_cuda.pt"
                    traced_cuda_b10 = torch.jit.trace(model, example_input_cuda_b10, strict=False)
                    traced_cuda_b10.save(str(gpu_output_path_b10))
                    print(f"✅ GPU版本模型(B=10)已保存到: {gpu_output_path_b10}")
                finally:
                    del traced_cuda
                    del traced_cuda_b10
                    del example_input_cuda
                    del example_input_cuda_b10
                    model.cpu()
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU版本模型序列化失败: {e}")

        print("复制并命名最佳模型文件...")
        new_model_path = output_dir / source_checkpoint_filename
        try:
            import shutil

            shutil.copyfile(checkpoint_path, new_model_path)
            print(f"✅ 最佳模型已复制到: {new_model_path}")
        except Exception as e:
            print(f"❌ 复制最佳模型失败: {e}")

        print("模型序列化完成！")
        return True
    except Exception as e:
        print(f"❌ 模型序列化失败: {e}")
        return False


def main():
    project_root = _setup_sys_path()
    default_mode = os.environ.get("AUDIO_CLASSIFY_DATA_MODE", "GM")
    args = _parse_args(default_mode)

    os.environ["AUDIO_CLASSIFY_DATA_MODE"] = args.data_mode
    import config  # noqa: E402

    classes_num = config.classes_num
    dataset_path = getattr(config, "dataset_path", "N/A")
    class_labels_file = getattr(config, "class_labels_file", "N/A")

    output_dir = _resolve_output_dir(project_root, config, args.output_dir)
    hardcoded = {
        "GM": project_root / "MobileNetV2_Mod_GM.pth",
        "DM": project_root / "MobileNetV2_Mod_DM.pth",
        "SEA": project_root / "MobileNetV2_Mod.pth",
    }
    checkpoint_path = args.checkpoint_path or str(hardcoded[args.data_mode])
    source_checkpoint_path = args.source_checkpoint_path or checkpoint_path

    print(f"Data mode: {args.data_mode}")
    print(f"Dataset path: {dataset_path}")
    print(f"Class labels file: {class_labels_file}")
    print(f"Classes num: {classes_num}")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint path: {checkpoint_path}")

    serialize_model(
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        mel_bins=args.mel_bins,
        classes_num=classes_num,
        output_dir=output_dir,
        source_checkpoint_path=source_checkpoint_path,
        example_input_len=args.example_input_len,
    )


if __name__ == "__main__":
    main()
