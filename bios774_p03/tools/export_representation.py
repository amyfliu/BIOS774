"""export_representation.py — save the RECOMMENDED representation as a compact deliverable.

  python tools/export_representation.py --data D --run O/<chosen model folder>

Writes <run's parent>/recommended_representation.npz (+ .json) with, for every
sample of the dataset:
  sample_id, split          — to join with data/<name>/obs.csv
  z_<modality> / z_joint    — the learned vectors, stored as float16 (half precision)

What is stored:
  * a multimodal model with one embedding per modality (clip, vicreg, identity,
    adapter, ...): the per-modality embeddings only. The joint embedding is
    derivable: joint = normalise(mean(normalise(z_a), normalise(z_b))), as train.py
    builds it — the .json says so.
  * a single-embedding model (pca_concat, simclr): z_joint.
float16 keeps ~3 significant digits, far more than cosine similarities and
linear probes need, and halves the file size. The full float32 embeddings of
every model stay in <model>/embeddings.npz (regenerable, not committed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, load_embeddings, load_json, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="the recommended model's folder, e.g. outputs/<run_id>/clip_mlp_d32")
    ap.add_argument("--out", default=None, help="default: <run's parent>/recommended_representation.npz")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    a = ap.parse_args()

    ds = Dataset(a.data)
    run = Path(a.run)
    emb = load_embeddings(run)
    cfg = load_json(run / "run_config.json")
    mod_keys = sorted(k for k in emb if k != "z_joint")
    keys = mod_keys if len(mod_keys) >= 2 else ["z_joint"]
    out = Path(a.out) if a.out else run.parent / "recommended_representation.npz"
    arrays = {k: emb[k].astype(a.dtype) for k in keys}
    for k, v in arrays.items():
        if len(v) != len(ds):
            raise SystemExit(f"{k} has {len(v)} rows, dataset has {len(ds)}")
    ids = np.array([str(x) for x in ds.ids], dtype=str)       # plain unicode arrays: load without pickle
    spl = np.array([str(x) for x in ds.split], dtype=str)
    np.savez_compressed(out, sample_id=ids, split=spl, **arrays)
    err = max(float(np.abs(arrays[k].astype(np.float32) - emb[k]).max()) for k in keys)
    info = {
        "dataset": ds.name, "source_model": run.name, "method": cfg.get("method"),
        "inputs": cfg.get("inputs"), "uses_labels": cfg.get("uses_labels", False),
        "keys": keys, "dims": {k: int(v.shape[1]) for k, v in arrays.items()},
        "n_samples": len(ds), "dtype": a.dtype, "max_abs_rounding_error": err,
        "rows": "same order as data/<name>/obs.csv; join on sample_id",
        "joint": ("joint = normalise(mean(normalise(z_a), normalise(z_b))) over the stored modality keys"
                  if keys != ["z_joint"] else "stored as z_joint"),
        "full_precision_source": str(run / "embeddings.npz"),
    }
    save_json(info, out.with_suffix(".json"))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {keys} from {run.name}, {a.dtype}, "
          f"max rounding error {err:.2g}")


if __name__ == "__main__":
    main()
