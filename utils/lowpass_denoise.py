#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import wave
from pathlib import Path
from typing import Tuple

import numpy as np


def _to_float_audio(raw_bytes: bytes, sampwidth: int, channels: int) -> np.ndarray:
    if sampwidth == 2:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if channels > 1:
        data = data.reshape(-1, channels)
    return data


def _from_float_audio(audio: np.ndarray, sampwidth: int) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    if sampwidth == 2:
        return (clipped * 32767.0).astype(np.int16).tobytes()
    if sampwidth == 4:
        return (clipped * 2147483647.0).astype(np.int32).tobytes()
    raise ValueError(f"Unsupported sample width: {sampwidth} bytes")


def lowpass_fft(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Low-pass filter via FFT by zeroing bins above cutoff."""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    n_samples = audio.shape[0]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(audio, axis=0)
    spectrum[freqs > cutoff_hz, :] = 0
    filtered = np.fft.irfft(spectrum, n=n_samples, axis=0)
    return filtered.astype(np.float32, copy=False)


def process_folder(input_dir: Path, cutoff_hz: float) -> Tuple[int, Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir = input_dir.parent / f"{input_dir.name}_denoise"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])

    if not audio_files:
        print(f"[WARN] No audio files found in: {input_dir}")
        return 0, output_dir

    for wav_path in audio_files:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            nframes = wf.getnframes()
            audio_raw = wf.readframes(nframes)

        audio = _to_float_audio(audio_raw, sampwidth, channels)
        filtered = lowpass_fft(np.asarray(audio), sample_rate, cutoff_hz)
        out_raw = _from_float_audio(filtered, sampwidth)

        with wave.open(str(output_dir / wav_path.name), "wb") as ow:
            ow.setnchannels(channels)
            ow.setsampwidth(sampwidth)
            ow.setframerate(sample_rate)
            ow.writeframes(out_raw)

    return len(audio_files), output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch low-pass denoise for folders.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more input directories containing audio files.",
    )
    parser.add_argument("--cutoff-hz", type=float, default=2500.0, help="Low-pass cutoff in Hz.")
    args = parser.parse_args()

    total = 0
    for in_dir in args.input_dirs:
        count, out_dir = process_folder(Path(in_dir), args.cutoff_hz)
        total += count
        print(f"[DONE] {in_dir} -> {out_dir} | files: {count} | cutoff: {args.cutoff_hz}Hz")

    print(f"[SUMMARY] processed files: {total}")


if __name__ == "__main__":
    main()
