# Domain-gap analysis — results

> **Note (correction):** real images carry an EXIF rotate-90 tag; the loader now
> applies `ImageOps.exif_transpose`. All numbers below are the corrected values.
> The mask-level headline (3D fragment gap > 2D) and the belt-removal
> decomposition both survived the fix. The per-class Spearman correlation was
> subsequently cut from the paper (underpowered, n=5).
>
> **Note (halo):** the 2D fragment library had its belt halo removed before
> compositing but the real SAM2 masks did not; the loader now peels the same
> blue edge from every mask (crops_from). After this, real and composited
> fragments are nearly indistinguishable at the mask level, and the 3D fragment
> gap is clear in the texture encoders (2.1x handcrafted, 1.7x DINOv2) but a
> near-tie for the YOLO detector backbone (1.06x, overlapping CIs).

Label-free measurement of the gap between the three training datasets (real /
2D-synthetic / 3D-synthetic), at three granularities, in three feature spaces.
Complements the metric-drop evidence already in the paper (Table 4).

## Design

| | |
|---|---|
| **Levels** | `scene` (whole image) · `bbox` (fragment + belt) · `mask` (fragment only) |
| **Feature spaces** | Handcrafted (color/texture/frequency) · DINOv2 ViT-S/14 · COCO-init YOLOv11s backbone |
| **Headline statistic** | MMD² (RBF, median-heuristic bandwidth) + permutation p, bootstrapped CIs |
| **Also reported** | energy distance, KID, proxy-A-distance (AUC), CORAL, hue-EMD |
| **Sampling** | scene: 26 real (all splits) vs 26 synth · instance: 300 (60/class) from the 6 **real test** scenes vs 300 matched synth |
| **Equal-N** | every comparison equal-N, synth subsampled, 300 bootstrap resamples |
| **Resolution** | all standardized to 1024² (the detector's input, §2.6); crops to 224² |

`bbox` and `mask` come from the **same** instances and differ **only** by whether
belt pixels are kept — so their difference isolates the background contribution.

Encoders are deliberately **neutral** (never trained on these domains), so each is
a fixed "ruler" (cf. Inception in FID). Three rulers that fail differently ⇒
agreement between them is stronger evidence than any single number.

## Finding 1 — a gap unambiguously exists (but AUC is saturated)

The domain-classifier **AUC is ~1.0 (0.996–1.000) in every space, level, and pair**; every MMD
permutation test gives p = 0.002. The domains are *trivially* separable
(t-SNE shows three disjoint clusters, `fig_tsne.png`), consistent with the
AUC ≈ 0.998 previously seen on the bitumen data.

⚠️ **Consequence:** proxy-A-distance/AUC **saturates** and cannot grade *how* large
a gap is. It supports only "a gap exists". All graded claims below rest on MMD
(and energy/KID, which agree).

## Finding 2 — at the fragment level, the 3D gap exceeds the 2D gap (ROBUST)

Mask-level (fragment-only) MMD²:

| Feature space | real↔2D | real↔3D | ratio |
|---|---|---|---|
| Handcrafted | 0.211 | 0.447 | 2.12× |
| DINOv2 | 0.089 | 0.150 | 1.69× |
| COCO-YOLO | 0.148 | 0.156 | 1.06× (tie) |

The two **texture-sensitive** rulers agree strongly: rendered fragments are
2.1× (handcrafted) and 1.7× (DINOv2) further from real than composited ones,
with non-overlapping bootstrap CIs. The **YOLO detector backbone is a near-tie**
(1.06×, overlapping CIs) — it keys on shape more than texture. Once the belt is
removed, real and composited fragments are nearly inseparable (AUC 0.997). This
is the quantitative form of the "texture deficit" argued in §2.5, and explains
why 3D-from-scratch (0.419 mAP) underperforms 2D (0.747).

## Finding 3 — the 2D gap is background-driven (PARTIALLY robust)

Handcrafted MMD² falls 0.96 → 0.70 → 0.21 (scene→bbox→mask) for real↔2D, but only
0.84 → 0.65 → 0.45 for real↔3D. So removing the belt **collapses** the 2D gap
(−78%) while leaving the 3D gap far less reduced (−47%). DINOv2 shows the same
ordering flip (2D above 3D at scene, below at mask).

⚠️ **Not fully robust:** COCO-YOLO ranks real↔3D *above* real↔2D at scene level
too, so the **scene-level ordering is encoder-dependent** and should not be
claimed. The fragment-level 3D>2D gap is clear only in the texture encoders
(Finding 2), not the YOLO backbone.

## Finding 4 — the gap predicts 3D's per-class failures, but not 2D's

Spearman ρ between per-class gap and per-class AP (expect negative):

| Feature space | vs 2D model AP | vs 3D Stage-1 AP |
|---|---|---|
| Handcrafted | +0.20 | −0.20 |
| **DINOv2 (mask)** | **−0.10** | **−0.80** (p = 0.104) |
| COCO-YOLO | +0.10 | −0.50 |

The DINOv2 fragment-level gap is near-monotonically predictive for the **purely
rendered** model: the two classes with the largest gap (AAC, Stones) are exactly
the two the 3D Stage-1 model fails on (AP 0.228 and 0.027), while the smallest-gap
classes (Tiles, Ceramics) score highest (`fig_gap_vs_ap.png`).

For **2D there is no relationship** (ρ = −0.10). This is a *positive* result, not
a null one: it says the 2D model's per-class variation is **not** gap-driven.
Consistent with §3.3, where 2D's errors are the AAC–mortar–stones confusion —
i.e. intrinsic class ambiguity, not domain shift.

**Interpretation:** 3D's weakness is a *domain-gap* problem (fixable by closing
the reality gap); 2D's weakness is a *class-separability* problem (fixable by more
fragments per class, as §3.2 already argues).

⚠️ **Caveat:** n = 5 classes ⇒ ρ = −0.80 gives p = 0.104, not significant at
α = 0.05 despite a strong effect size. Report as a strong trend with limited
power, not a confirmed correlation.

## Reproducing

```bash
python run_statistical.py   # handcrafted features -> outputs/statistical_results.json
SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())") \
REQUESTS_CA_BUNDLE=$SSL_CERT_FILE python run_latent.py   # -> outputs/latent_results.json
python run_figures.py       # figures + gap-vs-AP correlations
```

Features/embeddings cache to `cache/`; delete to force recompute. `run_latent.py`
takes ~20 min (kernel cost scales with the 384/512-d embeddings) — a worthwhile
optimization is to compute the pooled kernel once and re-index it for the
permutation/bootstrap loops instead of recomputing.
