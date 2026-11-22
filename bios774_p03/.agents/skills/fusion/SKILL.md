---
name: fusion
description: Choose how to combine the modalities (no fusion, early fusion, aligned shared space, supervised alignment, light fine-tuning of a pretrained model), the encoder, and what to compare it against.
---

# When to apply
After preprocessing, before choosing hyperparameters. Log one `fusion`
decision (strategy + baselines) and one `model_selection` decision (method +
encoder) — or a single `fusion` decision if they are inseparable.

# The options (train.py --method)

| Method | What it does | Lecture idea | Gives per-modality embeddings in one space? | Uses labels? |
|---|---|---|---|---|
| `pca_concat` (1 input) | single-modality baseline: PCA of one modality | PCA | — | no |
| `pca_concat` (all inputs) | early fusion: z-score, weight modalities equally, concatenate, PCA | PCA / early fusion | no (joint only) | no |
| `identity` | use a pretrained encoder's features as they are | foundation model, zero-shot | yes, if the encoder was trained jointly (e.g. BiomedCLIP image + text) | no |
| `clip` | one encoder per modality; paired samples pulled together, other pairs in the batch pushed apart (symmetric InfoNCE) | contrastive / CLIP (L23–L24) | yes | no |
| `vicreg` | one encoder per modality; paired embeddings close + each dimension keeps variance + dimensions decorrelated; no negatives | non-contrastive SSL (L24) | yes | no |
| `supcon_clip` | `clip` + supervised contrastive term: same-label samples are also positives | SupCon (L23) | yes | **yes** |
| `simclr` | ONE modality, two augmented views of each sample | SSL (L24) | — | no |

Encoders (`--encoder`): `mlp` (default for inputs from PCA/LSI/TF-IDF),
`linear` (tiny n, or to test whether nonlinearity matters), `adapter`
(residual MLP starting at the identity: light fine-tuning of pretrained
features; requires input dim = `--dim`, e.g. 512 for BiomedCLIP).

# Decision rules

**1. Always build the baselines** (objective skill): each modality alone and
`pca_concat` of all inputs; `identity` of the pretrained features when they
exist. They cost seconds.

**2. Pick the primary method from the objective and the data:**

| Situation (evidence) | Primary | Why / alternative worth one comparison |
|---|---|---|
| Paired samples, objective includes retrieval or cross-modal prediction, no pretrained shared-space encoder | `clip` with `mlp` encoders | a shared space is required; contrastive alignment is the standard way to get it. Alternative: `vicreg` |
| Same, but small n_train (< ~2,000) or many near-duplicate pairs (duplicate texts, cells of one type that look alike) | `vicreg` | contrastive loss treats look-alike pairs as negatives (false negatives) and is batch-size hungry; VICReg has no negatives. Alternative: `clip` with a smaller batch |
| A pretrained encoder already maps both modalities into one space (`identity` works: retrieval well above chance on val) | `clip` with `--encoder adapter` on the pretrained features (= light fine-tuning) | starts exactly at the zero-shot model, so it can only move if the data says so; cheap. Alternative: `mlp` projection heads on the same features (more capacity, more overfitting risk) |
| Objective is only integration/classification and modalities are unpaired or retrieval is irrelevant | `pca_concat` may already be enough — test `clip`/`vicreg` joint against it | a learned method is justified only if it beats early fusion |
| Objective explicitly allows labels for training | `supcon_clip` next to the label-free primary | report both; the label-free one answers "what do the modalities share?" |
| Only one modality has signal for the objective (e.g. single-modality baseline ≈ joint everywhere) | say so; keep the multimodal model only if it adds retrieval/prediction | |

**3. Embedding size** (`--dim`): ~ the number of meaningful directions —
16–64 for cell types (tens of classes), the pretrained dimension (512) for
an adapter. Bigger is not better for contrastive training at n ~10⁴.

**4. At most one alternative** to the primary in the first round; more only
as a logged iteration that a specific val result motivated.

# Pitfalls to check (and log if relevant)
- **False negatives** in contrastive training: many samples that genuinely
  match each other (same cell type; duplicate captions) are treated as
  negatives. Symptom: retrieval good within class, poor across. VICReg or a
  supervised term changes this — note the label policy.
- **Modality gap**: aligned embeddings can still sit in two separate clouds
  (`collapse.modality_gap_centroid_dist`); retrieval works anyway, but a joint
  embedding computed as their average then mixes two offsets — compare the
  joint vs per-modality probes.
- **Unequal modality strength**: if one modality alone matches the joint
  model, the second adds nothing for that task. That is a result.
