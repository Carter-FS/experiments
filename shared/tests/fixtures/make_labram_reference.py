"""Record reference outputs of the upstream (braindecode) LaBraM-base for the vendored copy.

Run in ``.venv-reve`` from the repository root:

    HF_HUB_OFFLINE=1 .venv-reve/bin/python shared/tests/fixtures/make_labram_reference.py

Writes ``labram_19ch_reference.npz`` beside this script: a fixed input ``x`` (2, 19, 2000)
in the LaBraM input convention, the mean-pooled feature ``mean`` and the [CLS] feature
``cls`` (2, 200) from ``shared.labram_pretrained.build_model``, the last transformer
block's output ``last_block`` (1, 191, 200) for the first sample, and a JSON ``meta``.
``shared/tests/test_vendored_labram.py`` requires the vendored model to reproduce these.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared import labram_pretrained as LP  # noqa: E402

OUT = Path(__file__).with_name("labram_19ch_reference.npz")


def main() -> None:
    import braindecode

    state = LP.hub_state_dict()
    mean_model, cls_model = LP.build_model("mean", state), LP.build_model("cls", state)
    x = torch.randn(2, LP.N_CHANNELS, LP.SAMPLES_PER_WINDOW, generator=torch.Generator().manual_seed(2026)) * 0.3
    captured = {}
    handle = mean_model.blocks[-1].register_forward_hook(lambda m, i, o: captured.setdefault("x", o.detach()))
    with torch.no_grad():
        mean = mean_model(x, ch_names=list(LP.LABRAM_CH_NAMES))
        cls = cls_model(x, ch_names=list(LP.LABRAM_CH_NAMES))
    handle.remove()
    meta = {
        "hub_repo": LP.HUB_REPO, "hub_revision": LP.HUB_REVISION, "hub_safetensors_sha256": LP.HUB_SAFETENSORS_SHA256,
        "braindecode_version": braindecode.__version__, "torch_version": torch.__version__,
        "labram_ch_names": list(LP.LABRAM_CH_NAMES), "model_kwargs": LP.MODEL_KWARGS, "input_chans": LP.input_chans(),
        "seed": 2026, "scale": 0.3,
    }
    np.savez_compressed(OUT, x=x.numpy(), mean=mean.numpy(), cls=cls.numpy(),
                        last_block=captured["x"][:1].numpy(), meta=json.dumps(meta, sort_keys=True))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
