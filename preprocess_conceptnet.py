#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess ConceptNet Numberbatch embeddings and align them with a given vocab.

Usage:
    python preprocess_conceptnet.py \
        --numberbatch_path /path/to/numberbatch-en-19.08.txt \
        --vocab_path /path/to/vocab.txt \
        --output_path knowledge_resources/conceptnet_features.pt

Expected formats:
- vocab.txt: one token per line, e.g.
    the
    a
    friend
    movie
- Numberbatch: space-separated text, e.g.
    word 0.123 0.456 ...
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


def parse_numberbatch_line(line: str):
    """
    Parse one line of Numberbatch-style embeddings.
    Example format:
        word 0.123 0.456 ...
    Returns: (token: str, vector: np.ndarray) or (None, None) if invalid.
    """
    parts = line.strip().split()
    if len(parts) < 10:
        # very short lines are likely headers or malformed
        return None, None
    token = parts[0]
    try:
        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    except ValueError:
        return None, None
    return token, vec


def load_numberbatch_embeddings(path: str, vocab_set) -> Dict[str, np.ndarray]:
    """
    Load only the embeddings whose token appears in vocab_set.
    This avoids loading the full (huge) ConceptNet file into memory.
    """
    embeddings = {}
    dim = None
    total_lines = 0
    matched = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            token, vec = parse_numberbatch_line(line)
            if token is None:
                continue
            if token in vocab_set:
                embeddings[token] = vec
                matched += 1
                if dim is None:
                    dim = vec.shape[0]

    print(f"[ConceptNet] Total lines read: {total_lines}")
    print(f"[ConceptNet] Matched vocab tokens: {matched}")
    if dim is None:
        raise RuntimeError("Could not infer embedding dimension. Check the file format.")
    print(f"[ConceptNet] Embedding dimension detected: {dim}")
    return embeddings


def build_embedding_matrix(
    vocab: List[str],
    embeddings: Dict[str, np.ndarray],
    emb_dim: int,
    init: str = "zeros",
) -> np.ndarray:
    """
    Build [vocab_size, emb_dim] matrix; OOV tokens are initialized by 'init'.

    init:
        'zeros' -> all zeros for OOV
        'normal' -> N(0, 0.02) random vectors for OOV
    """
    vocab_size = len(vocab)
    mat = np.zeros((vocab_size, emb_dim), dtype=np.float32)

    if init == "normal":
        mat = np.random.normal(loc=0.0, scale=0.02, size=(vocab_size, emb_dim)).astype(
            np.float32
        )

    covered = 0
    for idx, token in enumerate(vocab):
        if token in embeddings:
            vec = embeddings[token]
            if vec.shape[0] != emb_dim:
                # This should not happen if all vectors have same dim;
                # if it does, we skip or truncate.
                if vec.shape[0] > emb_dim:
                    vec = vec[:emb_dim]
                else:
                    # pad with zeros
                    tmp = np.zeros((emb_dim,), dtype=np.float32)
                    tmp[: vec.shape[0]] = vec
                    vec = tmp
            mat[idx] = vec
            covered += 1

    coverage = covered / vocab_size * 100.0
    print(f"[ConceptNet] Vocab size: {vocab_size}")
    print(f"[ConceptNet] Covered tokens: {covered} ({coverage:.2f}%)")

    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numberbatch_path", type=str, required=True,
                        help="Path to ConceptNet Numberbatch text file.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to vocab file (one token per line).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the resulting .pt tensor.")
    parser.add_argument("--oov_init", type=str, default="zeros",
                        choices=["zeros", "normal"],
                        help="Initialization for tokens not found in ConceptNet.")
    args = parser.parse_args()

    if not os.path.exists(args.numberbatch_path):
        print(f"Error: numberbatch file not found at {args.numberbatch_path}")
        sys.exit(1)
    if not os.path.exists(args.vocab_path):
        print(f"Error: vocab file not found at {args.vocab_path}")
        sys.exit(1)

    print("[ConceptNet] Loading vocab...")
    vocab = load_vocab(args.vocab_path)
    vocab_set = set(vocab)
    print(f"[ConceptNet] Vocab size: {len(vocab)}")

    print("[ConceptNet] Loading embeddings (filtered by vocab)...")
    embeddings = load_numberbatch_embeddings(args.numberbatch_path, vocab_set)

    # Infer embedding dim from one vector
    any_vec = next(iter(embeddings.values()))
    emb_dim = any_vec.shape[0]

    print("[ConceptNet] Building embedding matrix...")
    mat = build_embedding_matrix(
        vocab=vocab,
        embeddings=embeddings,
        emb_dim=emb_dim,
        init=args.oov_init,
    )

    tensor = torch.from_numpy(mat)  # [V, D]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(tensor, args.output_path)
    print(f"[ConceptNet] Saved tensor: {tensor.size()} → {args.output_path}")


if __name__ == "__main__":
    main()
