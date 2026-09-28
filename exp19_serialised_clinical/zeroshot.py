"""Zero-shot baseline for Experiment 19 (Addendum A.1): ask Llama-3.1-8B-Instruct directly.

For every Melbourne and HEP1 patient, the v1 text plus the pre-registered
question goes through the model's chat template (with the pre-registered
system message), and the score is P("Yes") / (P("Yes") + P("No")) from the
next-token distribution. No training, so there are no folds: one score per
patient, written to outputs/exp19_predictions/zeroshot_fp32_<cohort>.csv. The
two answer logits are computed in float32 from the final hidden state
(analysis plan B.7): the bfloat16 output layer quantised them to steps of
0.125 and left about 25 distinct scores for 198 patients.

    python -m exp19_serialised_clinical.zeroshot           # CPU, bfloat16, ~15 minutes
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.prediction_logger import run_provenance

from .config import PRED_DIR, ZERO_SHOT
from .texts import load_frames, texts

logger = logging.getLogger("exp19")


def answer_token_ids(tok) -> tuple[int, int]:
    """Single-token ids for the two answers as the assistant's first token."""
    yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise RuntimeError(f"'Yes'/'No' are not single tokens: {yes}, {no}")
    return yes[0], no[0]


@torch.inference_mode()
def score(text: str, tok, model, yes_id: int, no_id: int) -> float:
    messages = [{"role": "system", "content": ZERO_SHOT["system"]},
                {"role": "user", "content": f"{text}\n\n{ZERO_SHOT['question']}"}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=False)
    hidden = model.model(input_ids=ids).last_hidden_state[0, -1].float()
    weight = model.lm_head.weight[[yes_id, no_id]].float()
    pair = torch.softmax(weight @ hidden, dim=0)
    return float(pair[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="exp19 zero-shot Llama-3.1-8B-Instruct baseline")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    todo = [c for c in ("MEL", "HEP") if args.force or not (PRED_DIR / f"zeroshot_fp32_{c}.csv").exists()]
    if not todo:
        logger.info("zero-shot scores exist; skipping")
        return
    tok = AutoTokenizer.from_pretrained(ZERO_SHOT["model_id"], revision=ZERO_SHOT["revision"])
    model = AutoModelForCausalLM.from_pretrained(ZERO_SHOT["model_id"], revision=ZERO_SHOT["revision"],
                                                 dtype=torch.bfloat16).eval()
    yes_id, no_id = answer_token_ids(tok)
    for cohort in todo:
        frame = frames[(cohort, "full")]
        t0 = time.time()
        scores = []
        for i, text in enumerate(texts(frame, "v1")):
            scores.append(score(text, tok, model, yes_id, no_id))
            if (i + 1) % 100 == 0:
                logger.info(f"  {cohort} {i + 1}/{len(frame)} ({(time.time() - t0) / (i + 1):.2f}s per patient)")
        pd.DataFrame({"pid": frame["pid"].astype(str), "y_true": frame["outcome"].astype(int),
                      "score": scores}).to_csv(PRED_DIR / f"zeroshot_fp32_{cohort}.csv", index=False)
        logger.info(f"{cohort}: {len(frame)} patients in {time.time() - t0:.0f}s")
    (PRED_DIR / "zeroshot_fp32_manifest.json").write_text(json.dumps(
        {**ZERO_SHOT, "answer_logits": "float32 from the final hidden state", "provenance": run_provenance()},
        indent=2))


if __name__ == "__main__":
    main()
