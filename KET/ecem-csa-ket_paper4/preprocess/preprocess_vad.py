#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess NRC-VAD lexicon and align it with a given vocab.

Usage:
    python preprocess_vad.py \
        --vad_path /path/to/NRC-VAD-Lexicon.txt \
        --vocab_path /path/to/vocab.txt \
        --output_path knowledge_resources/vad_features.pt

Expected NRC-VAD format (tab-separated):
    word    valence arousal dominance
    abandon 0.19    0.41    0.44
    ...

We assume there is a header row.
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch


def load_vocab(vocab_path: str) -> List[str]:
    vocab = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                vocab.append(token)
    return vocab


def load_vad_lexicon(vad_path: str) -> Dict[str, np.ndarray]:
    """
    Load NRC-VAD lexicon into a dict: word -> [valence, arousal, dominance].
    Values are floats, usually between 0 and 1 or 1 and 9 depending on version.
    """
    vad_dict: Dict[str, np.ndarray] = {}
    with open(vad_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                # Some versions are tab-separated; try splitting on tab
                parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            word = parts[0]
            try:
                valence = float(parts[1])
                arousal = float(parts[2])
                dominance = float(parts[3])
            except ValueError:
                continue
            vad_dict[word] = np.array([valence, arousal, dominance], dtype=np.float32)
    print(f"[VAD] Loaded VAD entries: {len(vad_dict)}")
    return vad_dict


def build_vad_matrix(
    vocab: List[str],
    vad_dict: Dict[str, np.ndarray],
    default_value: float = 0.5,
) -> np.ndarray:
    """
    Build [vocab_size, 3] matrix.
    For tokens not found in VAD, assign a neutral default (e.g., 0.5).
    """
    vocab_size = len(vocab)
    mat = np.full((vocab_size, 3), default_value, dtype=np.float32)

    covered = 0
    for idx, token in enumerate(vocab):
        # NRC-VAD usually stores words in lowercase
        key = token.lower()
        if key in vad_dict:
            mat[idx] = vad_dict[key]
            covered += 1

    coverage = covered / vocab_size * 100.0
    print(f"[VAD] Vocab size: {vocab_size}")
    print(f"[VAD] Covered tokens: {covered} ({coverage:.2f}%)")

    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad_path", type=str, required=True,
                        help="Path to NRC-VAD lexicon file.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to vocab file (one token per line).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the resulting .pt tensor.")
    parser.add_argument("--default_value", type=float, default=0.5,
                        help="Default VAD value for OOV tokens (neutral).")
    args = parser.parse_args()

    if not os.path.exists(args.vad_path):
        print(f"Error: VAD file not found at {args.vad_path}")
        sys.exit(1)
    if not os.path.exists(args.vocab_path):
        print(f"Error: vocab file not found at {args.vocab_path}")
        sys.exit(1)

    print("[VAD] Loading vocab...")
    vocab = load_vocab(args.vocab_path)
    print(f"[VAD] Vocab size: {len(vocab)}")

    print("[VAD] Loading lexicon...")
    vad_dict = load_vad_lexicon(args.vad_path)

    print("[VAD] Building VAD feature matrix...")
    mat = build_vad_matrix(
        vocab=vocab,
        vad_dict=vad_dict,
        default_value=args.default_value,
    )

    tensor = torch.from_numpy(mat)  # [V, 3]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(tensor, args.output_path)
    print(f"[VAD] Saved tensor: {tensor.size()} → {args.output_path}")


if __name__ == "__main__":
    main()
