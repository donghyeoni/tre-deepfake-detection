# TRE Diffusion Image Detection

Detecting AI-generated images from the **Temporal Reconstruction Error (TRE)**
of a diffusion model, classified by temporal / spatial attention.

This repository holds the code, the **full-scale re-run of the experiment**, the
analysis of what the TRE feature measures under three reconstruction schemes,
and three follow-up experiments on scheme C.

## Results

GenImage. Train: SDv1.4, 30k fake + 30k real (seed 42), 20 epochs. Evaluation:
all 8 generators, full test splits (12k each, 16k for sdv5). Identical
classifier and hyperparameters in every condition — the only difference is how
the reconstruction is driven.

| Condition | Train acc | Held-out val acc | **Mean acc (8 generators)** |
| --- | --- | --- | --- |
| **A. Replayed noise** (as originally formulated) | 78.7% | 63.5% | **57.7%** |
| **B. Fresh common noise** (attempted fix) | 96.3% | 50.7% | **50.1%** |
| **C. Deterministic, η = 0** | — | 80.9% | **60.5%** |
| **C + second inverter** (SD1.4 ++ SD1.5, 8 ch) | — | 80.2% | **62.5%** |

Grouped by generator family:

| Grouped | Scheme C | C + second inverter |
| --- | --- | --- |
| Stable-Diffusion-derived (sdv4, sdv5, wukong) — same family as the inverter | **78.8%** | **78.9%** |
| Everything else (adm, biggan, glide, midjourney, vqdm) | **49.5%** | **52.7%** |

Per generator, raw numbers in [`results/repro.json`](results/repro.json),
[`results/fresh.json`](results/fresh.json) and [`results/eta0.json`](results/eta0.json).
The three SD-derived generators are marked ◆:

| | sdv4 ◆ (in-domain) | sdv5 ◆ | wukong ◆ | adm | biggan | glide | midjourney | vqdm | **mean** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: replayed — acc | 61.5 | 62.2 | 59.8 | 54.4 | 54.7 | 58.1 | 55.0 | 56.2 | **57.7** |
| A: replayed — AP | .642 | .648 | .623 | .541 | .560 | .593 | .552 | .562 | — |
| B: fresh — acc | 50.1 | 50.7 | 50.1 | 49.9 | 49.3 | 50.5 | 50.1 | 50.4 | **50.1** |
| B: fresh — AP | .503 | .504 | .505 | .499 | .490 | .504 | .499 | .503 | — |
| C: η = 0 — acc | **78.5** | **79.2** | **78.5** | 50.6 | 39.8 | 49.1 | 54.8 | 53.3 | **60.5** |
| C: η = 0 — AP | **.863** | **.869** | **.861** | .510 | .360 | .481 | .560 | .542 | — |
| C + 2nd inverter — acc | **79.3** | **79.2** | **78.2** | 52.9 | 45.5 | 53.8 | 55.3 | 55.9 | **62.5** |
| C + 2nd inverter — AP | **.870** | **.877** | **.862** | .538 | .429 | .549 | .567 | .576 | — |

## Why: the feature collapses either way

All three schemes share one skeleton — encode, walk the latent to noise,
reconstruct from every prefix, take the differences of consecutive
reconstructions. They differ only in the stochasticity coefficient `η` and in
what fills the noise slot of a reverse step:

![the three TRE schemes](assets/tre-schemes.svg)

The same input image, put through all three: the feature is a float-level
residue in A, pure noise in B, and structured reconstruction error in C.

![TRE feature under the three schemes](assets/tre-conditions.png)

*Input image: ImageNet ILSVRC2012 validation sample (real photograph),
distributed as the `nature` split of the GenImage benchmark. Heatmaps show the
channel mean of the TRE tensor; note the per-row scale annotation — A's range is
~10⁻⁴ while B's is ~1.*

The pipeline uses *edit-friendly* DDPM inversion, which records, for every step,
the noise `z_t` that makes the reverse process land exactly on the pre-sampled
`x_{t-1}`. TRE is then defined as the difference between reconstructions started
from different prefixes of that noise sequence.

- **Replaying `z_t`** (condition A) makes every prefix reconstruct the *same*
  latent by construction, so the difference is mathematically zero. What remains
  is GPU floating-point non-determinism: measured `std ~ 2.7e-4` against a latent
  scale of ~1. The weak in-domain signal is that residue's image-dependent
  pattern, which is why it does not transfer across generators.
- **Injecting fresh noise** (condition B) makes prefixes genuinely differ, but
  the variance term `sigma_t * eps_t` dominates: prefixes of different length
  accumulate a different number of such terms, so the difference is driven by the
  image's random draw rather than by how well the model explains the image. The
  classifier memorises the draw (96% train) and transfers nothing.

Either way the stochastic reverse process destroys the quantity the method
intends to measure. Full derivation, measurements and follow-up directions:
[`docs/finding-tre-collapse.md`](docs/finding-tre-collapse.md) and
[`docs/follow-up.md`](docs/follow-up.md).

**Scheme C (`η = 0`) — the feature works, the generalisation does not.**
Dropping the stochastic term leaves the prefix differences to reflect only DDIM
inversion error, i.e. how accurately the model round-trips the image. The feature
magnitude rises ~500x over scheme A (std 0.20 vs 2·10⁻⁴), and real photographs
carry a ~17% larger error than generated ones — the direction DIRE/LaRE-style
detectors rely on. At full scale that turns into a genuine detector for images
made by the *same model family as the inverter* (SD v1.4): 78-79% accuracy at
AP 0.86 on sdv4, sdv5 and wukong, and the held-out validation accuracy reaches
80.9%.

Outside that family it collapses to chance, and on biggan it inverts (39.8%,
AP 0.36) — a GAN's latents are not something an SD inverter round-trips the way
it does its own samples, so the learned direction points the wrong way. So the
three schemes fail for three different reasons: A has no signal, B has signal
buried in noise, C has signal that is specific to one generator family.

## Follow-up experiments on scheme C

Three experiments on why scheme C is family-bound, each with its own file in
[`results/`](results).

**Leave-one-generator-out** ([`logo.json`](results/logo.json)). Train on five
generators' test-half features (first half of each class, 30k), test on the
sixth, six times; same architecture and hyperparameters.

| Held-out generator | Trained on | Val acc (trained gens) | Held-out acc | Held-out AP |
| --- | --- | --- | --- | --- |
| adm | biggan, glide, midjourney, vqdm, wukong | 53.1 | 52.3 | 0.527 |
| biggan | adm, glide, midjourney, vqdm, wukong | 54.0 | 53.0 | 0.534 |
| glide | adm, biggan, midjourney, vqdm, wukong | 51.8 | 53.0 | 0.535 |
| midjourney | adm, biggan, glide, vqdm, wukong | 53.4 | 53.4 | 0.538 |
| vqdm | adm, biggan, glide, midjourney, wukong | 58.2 | 48.0 | 0.488 |
| wukong | adm, biggan, glide, midjourney, vqdm | 52.8 | 50.4 | 0.504 |
| **mean** | | | **51.7** | |

Held-out accuracy is 48.0-53.4% in every split, and validation on the five
*trained* generators is 51.8-58.2%. wukong falls from 78.5% (sdv1.4-only
training) to 50.4% once its training set is the five non-SD generators: the
family bias is a property of the η = 0 feature, not of the training mix.

**3-class head** ([`threeclass.json`](results/threeclass.json)). Real /
diffusion-fake / GAN-fake, trained on the first half of each of the six test
generators' features (36k; biggan is the only GAN); binary accuracy collapses
the two fake classes. Per-generator numbers are over *both* halves, i.e. they
include the training half; the held-out binary accuracy (second halves only) is
54.3%.

| Generator | 3-class acc | Binary acc | AP | Scheme C binary acc |
| --- | --- | --- | --- | --- |
| adm | 58.4 | 60.0 | 0.628 | 50.6 |
| biggan | 44.2 | 63.4 | 0.683 | 39.8 |
| glide | 56.8 | 60.8 | 0.643 | 49.1 |
| midjourney | 57.7 | 58.6 | 0.617 | 54.8 |
| vqdm | 57.2 | 57.3 | 0.601 | 53.3 |
| wukong | 57.2 | 57.4 | 0.592 | 78.5 |

biggan's below-chance result under scheme C (39.8%) becomes 63.4% / AP 0.68 once
GAN-fake is its own class.

**Two-inverter ensemble** ([`ensemble.json`](results/ensemble.json)). SD 1.4
and SD 1.5 features concatenated on the channel axis, `(T=20, 8, 32, 32)`,
trained on the sdv1.4 split (60k). Per-generator numbers are in the main table
above; the single-inverter controls:

| | Ensemble (8 ch) | SD 1.4 only | SD 1.5 only |
| --- | --- | --- | --- |
| Val acc | 80.2 | 80.9 | 79.8 |
| sdv4 acc | 79.3 | 78.5 | 78.7 |
| sdv5 acc | 79.2 | 79.2 | 78.7 |

SD family 78.8% → 78.9%, other generators 49.5% → 52.7%, biggan 39.8% → 45.5%.
SD 1.5 is a continuation of SD 1.4, so both inverters are the same family (SD 2.1
was the intended second inverter; `stabilityai/*` repositories are gated).

Open hypotheses and next directions are collected in
[`docs/follow-up.md`](docs/follow-up.md).

Baseline numbers (STRE, NPR, DIRE, LaRE) are **not reproduced here**; cite them
from their original papers, noting protocol differences.

## Code layout

```
├── src/                  # library: inversion, TRE features, datasets, models
├── experiments/          # list building, TRE extraction, trainers (entry points)
├── scripts/              # server bootstrap and multi-GPU shard / stage runners
├── results/              # measured accuracy/AP per generator, every experiment
└── docs/                 # analysis of the collapse, reproduction guide, follow-up, project report
```

The experiment scripts import `src/config.py`, `src/data/inversion.py` and
`src/models/resnet_baseline.py`; `experiments/extract_tre.py` is the batched
port of `src/data/tre_features.py`. The remaining `src/` modules (`train.py`,
`eval.py`, `data/build_dataset.py`, `data/dataset.py` and the DNSAMNet models)
are the original notebook pipeline and are not called by the experiments.

## Reproducing

Data (GenImage), features and weights are not distributed. Environment, data
preparation, extraction cost and every run command:
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md); bugs fixed in the original code:
its section 7.
