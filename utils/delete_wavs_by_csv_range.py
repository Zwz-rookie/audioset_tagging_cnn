import argparse
import csv
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple


class CsvLineAudioRef(object):
    def __init__(self, line_number, audio_name, wav_path):
        self.line_number = int(line_number)
        self.audio_name = str(audio_name)
        self.wav_path = wav_path


def _infer_audio_dir_from_csv(csv_path: Path) -> Path:
    dataset_root = csv_path.resolve().parent.parent
    candidates = [
        dataset_root / "audios" / "balanced_train_segments_GM",
        dataset_root / "audio" / "balanced_train_segments_GM",
        dataset_root / "audios",
        dataset_root / "audio",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _sample_audio_names_in_range(
    csv_path: Path, start_line: int, end_line: Optional[int], limit: int
) -> List[str]:
    names = []  # type: List[str]
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",", skipinitialspace=True)
        for line_number, row in enumerate(reader, start=1):
            if line_number < start_line:
                continue
            if end_line is not None and line_number > end_line:
                break
            if not row:
                continue
            audio_name = str(row[0]).strip()
            if audio_name.lower() == "audio_name":
                continue
            names.append(audio_name)
            if len(names) >= limit:
                break
    return names


def _auto_detect_audio_dir(
    csv_path: Path, start_line: int, end_line: Optional[int]
) -> Tuple[Path, int, int]:
    dataset_root = csv_path.resolve().parent.parent
    candidates = [
        dataset_root / "audios" / "balanced_train_segments_GM",
        dataset_root / "audio" / "balanced_train_segments_GM",
        dataset_root / "audios",
        dataset_root / "audio",
    ]

    sample = _sample_audio_names_in_range(csv_path, start_line, end_line, limit=50)
    best_dir = None  # type: Optional[Path]
    best_hits = -1

    for cand in candidates:
        if not cand.exists():
            continue
        hits = 0
        for audio_name in sample:
            rel_wav_path = _audio_name_to_rel_wav_path(audio_name)
            if (cand / rel_wav_path).exists():
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_dir = cand

    if best_dir is not None:
        return best_dir, best_hits, len(sample)

    inferred = _infer_audio_dir_from_csv(csv_path)
    return inferred, 0, len(sample)


def _audio_name_to_rel_wav_path(audio_name: str) -> Path:
    audio_name = audio_name.strip().strip("\ufeff")
    rel = Path(*PurePosixPath(audio_name).parts)
    if rel.suffix == "":
        rel = rel.with_suffix(".wav")
    return rel


def iter_audio_refs(
    csv_path: Path,
    audio_dir: Path,
    start_line: int,
    end_line: Optional[int],
) -> Iterable[CsvLineAudioRef]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",", skipinitialspace=True)
        for line_number, row in enumerate(reader, start=1):
            if line_number < start_line:
                continue
            if end_line is not None and line_number > end_line:
                break
            if not row:
                continue
            audio_name = str(row[0]).strip()
            if audio_name.lower() == "audio_name":
                continue
            rel_wav_path = _audio_name_to_rel_wav_path(audio_name)
            wav_path = audio_dir / rel_wav_path
            yield CsvLineAudioRef(
                line_number=line_number, audio_name=audio_name, wav_path=wav_path
            )


def delete_wavs(
    refs: Iterable[CsvLineAudioRef],
    dry_run: bool,
) -> int:
    unique_paths = {}  # type: Dict[Path, CsvLineAudioRef]
    for ref in refs:
        unique_paths.setdefault(ref.wav_path, ref)

    missing = []  # type: List[CsvLineAudioRef]
    deleted = []  # type: List[CsvLineAudioRef]
    errors = []  # type: List[Tuple[CsvLineAudioRef, Exception]]

    for wav_path, ref in unique_paths.items():
        if not wav_path.exists():
            missing.append(ref)
            continue

        if dry_run:
            deleted.append(ref)
            continue

        try:
            wav_path.unlink()
            deleted.append(ref)
        except Exception as e:
            errors.append((ref, e))

    print(f"Matched unique wavs: {len(unique_paths)}")
    print(f"Exists: {len(unique_paths) - len(missing)}")
    print(f"Missing: {len(missing)}")
    print(f"{'Would delete' if dry_run else 'Deleted'}: {len(deleted)}")
    print(f"Errors: {len(errors)}")

    if missing:
        print("\nMissing examples (up to 20):")
        for ref in missing[:20]:
            print(f"  line {ref.line_number}: {ref.wav_path}")

    if errors:
        print("\nError examples (up to 20):")
        for ref, e in errors[:20]:
            print(f"  line {ref.line_number}: {ref.wav_path} -> {type(e).__name__}: {e}")

    return 0 if not errors else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--start-line", type=int, required=True)
    parser.add_argument("--end-line", type=int, default=None)
    parser.add_argument("--audio-dir", type=str, default=None)
    parser.add_argument("--delete", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.audio_dir is None:
        audio_dir, hits, total = _auto_detect_audio_dir(csv_path, args.start_line, args.end_line)
        if total > 0:
            print(f"Auto-detected audio dir hits: {hits}/{total}")
    else:
        audio_dir = Path(args.audio_dir)

    start_line = int(args.start_line)
    end_line = int(args.end_line) if args.end_line is not None else None

    if start_line < 1:
        raise ValueError("--start-line must be >= 1")
    if end_line is not None and end_line < start_line:
        raise ValueError("--end-line must be >= --start-line")

    dry_run = not bool(args.delete)

    refs = list(iter_audio_refs(csv_path, audio_dir, start_line, end_line))
    if not refs:
        print("No rows matched the specified line range.")
        return 0

    print(f"CSV: {csv_path.resolve()}")
    print(f"Audio dir: {audio_dir.resolve()}")
    print(f"Line range: {start_line}..{end_line if end_line is not None else 'EOF'}")
    print(f"Mode: {'DELETE' if not dry_run else 'DRY-RUN'}")
    return delete_wavs(refs, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
