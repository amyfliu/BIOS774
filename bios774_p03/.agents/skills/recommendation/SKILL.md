---
name: recommendation
description: Run and report the downstream analyses the objective asked for, then give one consistent recommendation of which representation to use, for what, and with which caveats.
---

# When to apply
After the test evaluation of the chosen model(s) and baselines.

# Downstream analyses
The objective names them (objective skill); each gets its own report
subsection with: the question in plain words, the method in one sentence
(e.g. "a linear classifier trained on the embeddings of training cells"),
the headline number next to its baseline and chance, the 95% CI, and one
sentence of interpretation. At least one downstream analysis is required.

| Downstream task | Evidence to show |
|---|---|
| classification | probe balanced accuracy (+ macro-F1) for chosen model vs every baseline; the 2–3 classes it handles best/worst (`probe_per_class_recall`) |
| clustering | ARI/NMI vs baselines; the embedding figure |
| retrieval | recall@1/5/10 both directions vs chance and vs zero-shot; one sentence on what a miss tends to look like if visible (e.g. same modality, different body part) |
| cross-modal prediction | per-sample r / AUROC vs the train-mean baseline and vs the single-modality baseline embedding; per-feature r and the fraction of features predicted with r > 0.3 |

Log one `downstream` decision summarising the analyses run and their headline
numbers (with file names).

# The recommendation (log `recommendation`)
Answer the objective in one sentence, then:

1. **Which representation, for which use.** It can differ by task — e.g.
   "joint VICReg embedding for cell-type annotation; the RNA encoder alone
   when only RNA is measured" or "fine-tuned adapter for retrieval; the
   zero-shot model is within its CI for modality classification".
2. **The deciding numbers**: the chosen model vs the best baseline on the
   primary metric, as a PAIRED difference with its 95% CI
   (`comparison_test_vs_<baseline>.md`), next to the two values.
3. **Verdict categories** (pick one, use it in the header, Summary and
   Recommendation):
   - *Multimodal helps*: the chosen multimodal representation beats the best
     single-modality / zero-shot baseline on the primary metric, with a paired
     difference whose CI excludes 0 and a size that matters.
   - *Multimodal enables*: no gain on classification, but it provides
     something a single modality cannot (retrieval, cross-modal prediction).
   - *No measurable gain*: the paired difference vs the best baseline has a CI
     that includes 0 (or is too small to matter) — recommend the simpler
     baseline (fewer moving parts), and say what evidence would change this.
   - *Can't tell*: the test set cannot resolve the question (too few samples
     in the relevant classes); say which data would.
4. **Caveats that change how to use it**: label provenance (labels predicted
   from one modality), text shortcuts, label-trained models, duplicates,
   subset size, one donor / one data source, and **pretraining overlap**: if
   a pretrained encoder was trained on data from the same source as this
   dataset (compare `DATA.md` source with what is known about the encoder's
   pretraining data), its zero-shot scores — and anything built on them — may
   be inflated by having seen test samples. State it as a possibility unless
   the overlap can be ruled out; it cannot be measured with these tools.
5. **What would change the recommendation** (one or two concrete items).

Implementation matters: prefer the simplest representation whose paired
difference from the best includes 0 — a frozen pretrained model or PCA is
easier to deploy and audit than a trained encoder.

# Naming numbers consistently
A model with several embeddings (joint, plus one per modality) yields
several "RNA" numbers. In the Summary, Results and Recommendation, "RNA
alone" / "image alone" always means the single-modality BASELINE model. If
you also discuss a multimodal model's own per-modality embedding (e.g. its
RNA encoder), name it explicitly ("the joint model's RNA encoder") and never
put it where a baseline number is expected. The comparison table and the
prose must show the same numbers for the same names.
