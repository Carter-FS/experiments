"""Compute and cache the exp19 text embeddings (Addendum A / A.1 of the analysis plan).

Collects every distinct text any configuration, fold or seed will use
(texts.all_texts), embeds each once per encoder at its pinned revision and
stores it under the SHA-256 of the text:

    outputs/exp19_embeddings/<enc>_store.keys.txt      one text key per row
    outputs/exp19_embeddings/<enc>_store_<pool>.npy    float32 (n_texts, dim), rows match the keys
    outputs/exp19_embeddings/texts_store.csv           key, text (patient-level)
    outputs/exp19_embeddings/manifest_<enc>.json       revision, pooling, truncation, counts

Reruns only embed texts missing from a store. outputs/ is gitignored.

    python -m exp19_serialised_clinical.embed --encoders pubmedbert clinicalbert
    python -m exp19_serialised_clinical.embed --encoders llama31_8b          # CPU, bfloat16
    python -m exp19_serialised_clinical.embed --encoders qwen3_embed_8b      # CPU, bfloat16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from shared.prediction_logger import run_provenance
from shared.serialise_clinical import TEMPLATE_VERSION, text_key

from .config import EMB_DIR, ENCODERS, POOLINGS, QWEN_INSTRUCTION, SEEDS
from .texts import all_texts, load_frames

logger = logging.getLogger("exp19")


def load_tokenizer(encoder: str):
    spec = ENCODERS[encoder]
    tok = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
    if tok.pad_token is None:  # Llama has none; it is only ever used for masked positions
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


DECODER_KINDS = ("decoder", "embed_lasttok")


def load_model(encoder: str, device: torch.device):
    spec = ENCODERS[encoder]
    kwargs = {"dtype": torch.bfloat16} if spec["kind"] in DECODER_KINDS else {}
    return AutoModel.from_pretrained(spec["model_id"], revision=spec["revision"], **kwargs).to(device).eval()


@torch.inference_mode()
def embed_bert(texts: list[str], tok, model, device, batch_size: int, max_len: int):
    """Mask-aware mean pooling; texts longer than max_len are split into windows of
    max_len - 2 tokens (each wrapped in [CLS] ... [SEP]) and the window means are
    averaged weighted by window length. Returns ({"mean": array}, n_windowed)."""
    width = max_len - 2
    windows, n_windowed = [], 0
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_windowed += len(ids) > width
        for start in range(0, max(len(ids), 1), width):
            windows.append((i, [tok.cls_token_id] + ids[start:start + width] + [tok.sep_token_id]))
    sums = np.zeros((len(texts), model.config.hidden_size), dtype=np.float64)
    weights = np.zeros(len(texts))
    order = sorted(range(len(windows)), key=lambda j: len(windows[j][1]))
    for b in range(0, len(order), batch_size):
        chunk = [windows[j] for j in order[b:b + batch_size]]
        length = max(len(w) for _, w in chunk)
        ids = torch.full((len(chunk), length), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), length), dtype=torch.long)
        for r, (_, w) in enumerate(chunk):
            ids[r, :len(w)] = torch.tensor(w)
            mask[r, :len(w)] = 1
        hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state.float()
        m = mask.to(device).unsqueeze(-1).float()
        means = ((hidden * m).sum(1) / m.sum(1)).cpu().numpy()
        for r, (i, w) in enumerate(chunk):
            sums[i] += means[r] * len(w)
            weights[i] += len(w)
    return {"mean": (sums / weights[:, None]).astype(np.float32)}, n_windowed


@torch.inference_mode()
def embed_decoder(texts: list[str], tok, model, device, max_len: int):
    """One text per forward pass (no padding). Mean over final hidden states excluding
    BOS (an attention sink with very large activations), and the last token's state.
    Returns ({"mean": array, "last": array}, n_truncated)."""
    dim = model.config.hidden_size
    mean, last = np.zeros((len(texts), dim), np.float32), np.zeros((len(texts), dim), np.float32)
    n_truncated, t0 = 0, time.time()
    for i, text in enumerate(texts):
        ids = tok(text)["input_ids"]
        if len(ids) > max_len:
            ids, n_truncated = ids[:max_len], n_truncated + 1
        hidden = model(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0].float()
        start = 1 if ids[0] == tok.bos_token_id else 0
        mean[i] = hidden[start:].mean(0).cpu().numpy()
        last[i] = hidden[-1].cpu().numpy()
        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(texts)} texts ({(time.time() - t0) / (i + 1):.2f}s per text)")
    return {"mean": mean, "last": last}, n_truncated


@torch.inference_mode()
def embed_lasttok(texts: list[str], tok, model, device, max_len: int):
    """Embedding-tuned decoder (Qwen3-Embedding): the model card's instruction
    prefix, one text per pass, the final token's state (the end-of-text token,
    appended when the tokenizer does not add it), L2 normalised. Returns
    ({"last": array}, n_truncated)."""
    dim = model.config.hidden_size
    last = np.zeros((len(texts), dim), np.float32)
    eos = tok.eos_token_id
    n_truncated, t0 = 0, time.time()
    for i, text in enumerate(texts):
        ids = tok(QWEN_INSTRUCTION + text)["input_ids"]
        if ids[-1] != eos:
            ids = ids + [eos]
        if len(ids) > max_len:
            ids, n_truncated = ids[:max_len - 1] + [eos], n_truncated + 1
        hidden = model(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0, -1].float()
        last[i] = torch.nn.functional.normalize(hidden, dim=0).cpu().numpy()
        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(texts)} texts ({(time.time() - t0) / (i + 1):.2f}s per text)")
    return {"last": last}, n_truncated


def load_store(out_dir: Path, encoder: str) -> tuple[list[str], dict[str, np.ndarray]]:
    keys_path = out_dir / f"{encoder}_store.keys.txt"
    if not keys_path.exists():
        return [], {}
    keys = keys_path.read_text().split()
    arrays = {p: np.load(out_dir / f"{encoder}_store_{p}.npy") for p in POOLINGS[ENCODERS[encoder]["kind"]]}
    for p, arr in arrays.items():
        if len(arr) != len(keys):
            raise RuntimeError(f"{encoder} store is inconsistent ({p}: {len(arr)} rows, {len(keys)} keys)")
    return keys, arrays


def save_store(out_dir: Path, encoder: str, keys: list[str], arrays: dict[str, np.ndarray]) -> None:
    # Arrays first, keys last: a crash never leaves keys pointing past the arrays.
    for p, arr in arrays.items():
        tmp = out_dir / f"{encoder}_store_{p}.tmp.npy"
        np.save(tmp, arr)
        tmp.replace(out_dir / f"{encoder}_store_{p}.npy")
    tmp = out_dir / f"{encoder}_store.keys.tmp"
    tmp.write_text("\n".join(keys) + "\n")
    tmp.replace(out_dir / f"{encoder}_store.keys.txt")


POOLING_NOTES = {
    "decoder": "mean over final hidden states excluding BOS; last = final token's state",
    "embed_lasttok": "instruction prefix; final (end-of-text) token's state, L2 normalised",
    "bert": "mask-aware mean; >512 tokens: 510-token windows, length-weighted average of window means",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="exp19: embed every serialised text once per encoder")
    parser.add_argument("--encoders", nargs="+", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--device", default=None, help="default: cuda for BERT models if available; cpu for Llama")
    parser.add_argument("--batch-size", type=int, default=32, help="BERT windows per forward pass")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="seeds whose fold fills are needed")
    parser.add_argument("--out-dir", type=Path, default=EMB_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    needed = all_texts(load_frames(), seeds=args.seeds)
    pd.DataFrame({"key": [text_key(t) for t in needed], "text": needed}).to_csv(
        args.out_dir / "texts_store.csv", index=False)
    logger.info(f"{len(needed)} distinct texts")

    for encoder in args.encoders:
        spec = ENCODERS[encoder]
        decoder = spec["kind"] in DECODER_KINDS
        device = torch.device(args.device or ("cpu" if decoder or not torch.cuda.is_available() else "cuda"))
        keys, arrays = load_store(args.out_dir, encoder)
        have = set(keys)
        todo = [t for t in needed if text_key(t) not in have]
        if not todo:
            logger.info(f"{encoder}: store complete ({len(keys)} texts)")
            continue
        logger.info(f"{encoder}: embedding {len(todo)} new texts with {spec['model_id']}@{spec['revision'][:8]} on {device}")
        tok, model = load_tokenizer(encoder), load_model(encoder, device)
        t0 = time.time()
        if spec["kind"] == "embed_lasttok":
            new, n_cut = embed_lasttok(todo, tok, model, device, spec["max_len"])
        elif decoder:
            new, n_cut = embed_decoder(todo, tok, model, device, spec["max_len"])
        else:
            new, n_cut = embed_bert(todo, tok, model, device, args.batch_size, spec["max_len"])
        keys = keys + [text_key(t) for t in todo]
        arrays = {p: np.concatenate([arrays[p], new[p]]) if p in arrays else new[p] for p in new}
        save_store(args.out_dir, encoder, keys, arrays)
        manifest_path = args.out_dir / f"manifest_{encoder}.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
        manifest.update({"encoder": encoder, "model_id": spec["model_id"], "revision": spec["revision"],
                         "dtype": "bfloat16" if decoder else "float32", "max_len": spec["max_len"],
                         "template_version": TEMPLATE_VERSION, "n_texts": len(keys),
                         "pooling": POOLING_NOTES[spec["kind"]]})
        manifest["runs"].append({"n_new": len(todo), ("n_truncated" if decoder else "n_windowed"): int(n_cut),
                                 "seconds": round(time.time() - t0, 1), "provenance": run_provenance()})
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"{encoder}: {len(todo)} texts in {time.time() - t0:.0f}s "
                    f"({'truncated' if decoder else 'windowed'}: {n_cut})")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
