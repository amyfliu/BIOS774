# ROCOv2 — radiology images + captions

**Samples:** 24000 figure images from open-access biomedical papers
(PubMed Central), each paired with its **figure caption**, so row *i* is one
image–text pair (paired multimodal data).

| modality | kind | stored as |
|---|---|---|
| `image` | image | JPEG, RGB, shorter side resized to 256 px (aspect kept), path in column `image_path` |
| `text` | text | raw caption in column `caption` (median 17 words, 90th pct 38) |

Most images are grayscale scans stored as RGB. Captions are written by paper
authors for readers, not clinical reports: they often name the imaging
modality, body part and finding, and can mention arrows or panel letters.

## Labels (for evaluation)

- `label`: imaging modality, from ROCOv2's manually curated concepts
  (checked by a radiologist). Images with no or several modality concepts were
  dropped (see below).
- `cuis` / `concept_names`: all automatically extracted UMLS concepts of the
  caption (multi-label; usable as extra evaluation targets).

| label | train | val | test | total |
|---|---:|---:|---:|---:|
| CT | 6998 | 791 | 717 | 8506 |
| X-ray | 5581 | 543 | 565 | 6689 |
| MRI | 3299 | 276 | 301 | 3876 |
| Ultrasound | 2746 | 283 | 322 | 3351 |
| Angiography | 1263 | 98 | 82 | 1443 |
| PET | 113 | 9 | 13 | 135 |

## Splits and subset

ROCOv2's own split (train / valid → `val` / test) is kept, so `test` images
come from different papers than `train`. A fixed random subset (seed 0)
of each split is used to keep training laptop-sized:

| split | available with one modality label | kept |
|---|---:|---:|
| train | 59465 | 20000 |
| val | 9773 | 2000 |
| test | 9822 | 2000 |

Fit every preprocessing step and model on `train` only; use `val` for model
choices and `test` for the final numbers.

## Modality concepts used for the label

| CUI | concept | label | images |
|---|---|---|---|
| C0040405 | X-Ray Computed Tomography | CT | 27747 |
| C1306645 | Plain x-ray | X-ray | 21997 |
| C0024485 | Magnetic Resonance Imaging | MRI | 12657 |
| C0041618 | Ultrasonography | Ultrasound | 11429 |
| C0002978 | angiogram | Angiography | 4799 |
| C0032743 | Positron-Emission Tomography | PET | 432 |

## Preparation already applied (not the agent's choice)

- Resizing (shorter side 256 px) and RGB conversion only; no cropping,
  no normalisation, no caption cleaning or tokenisation.

Source: https://zenodo.org/records/10821435 (CC BY-NC 4.0; non-commercial use only; individual images
keep the licences listed in ROCOv2's `license_information.csv`).
