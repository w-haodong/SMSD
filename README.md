# Structured Multiscale Sparse Decoding for Vertebral Point Detection in Scoliosis Radiographs

This repository provides the training, validation, inference, and held-out test evaluation code for our structured multiscale sparse decoding framework for vertebral point detection in scoliosis radiographs.

## Abstract

Accurate Cobb angle measurement is fundamental to scoliosis assessment and follow-up. However, standing radiographs acquired in clinical practice often retain broad anatomical coverage from the skull and shoulders to the pelvis and lower limbs. Non-spinal osseous structures may consequently produce local responses similar to those of vertebrae, destabilizing vertebral localization and subsequent angle computation. A structured multiscale sparse decoding framework is proposed for vertebral localization and Cobb angle measurement in scoliosis radiographs. The framework uses vertebral point detection as its geometric output form, introduces a centerline-guided continuous spinal-structure reference during feature decoding, and constructs multiscale vertebra-level structural features. Spinal arrangement context is combined with high-resolution local geometry to predict vertebral centers, center offsets, and corner offsets. By interpreting local osseous responses within the continuous spinal structure, the framework improves the recognition stability of vertebral structural units. The proposed method achieves 6.09% SMAPE and 1.54 deg MAE on the AASCE test set, and 5.92% overall SMAPE, 2.07 deg overall MAE, and 4.19 px 68-corner Point ED on the Clinical-150 external clinical test set, demonstrating stable vertebral point localization and Cobb angle measurement.

## Framework

![Structured multiscale sparse decoding framework](assets/framework.jpg)

Structured multiscale sparse decoding framework. The model combines deep-context feature gating, centerline-guided structural prior learning, centerline-guided coarse localization, vertebra-level chain refinement, and high-resolution candidate gathering before decoding vertebral points and Cobb-related geometry.

## Method-to-Code Map

The paper presents the structural formulation of SMSD, while this repository records the implementation-level choices needed to reproduce the released pipeline.

| Method component | Main implementation |
| --- | --- |
| Multiscale backbone and decoder assembly | `models/DecNet.py`: `StructuredSpineDecoder` |
| Global semantic calibration | `models/DecNet.py`: `SemanticP5Bias` |
| Centerline parameterization and row-level structural propagation | `models/DecNet.py`: `ExplicitAxisRepresentation` |
| Narrow/wide centerline-aligned corridors and vertebral-chain reasoning | `models/DecNet.py`: `CanonicalStripContext` |
| High-resolution sparse candidate refinement | `models/DecNet.py`: `PeakOffsetRefiner` |
| Heatmap and offset decoding | `models/DecNet.py`, `operation/decode.py`, and `main.py` |
| Centerline and vertebral supervision targets | `datasets/dataset.py` |
| Training objectives | `operation/loss.py` |
| Training and validation loop | `operation/train.py` |

The canonical release profile is defined in `main.py::_apply_internal_profile`. Its principal decoder settings are:

| Setting | Value |
| --- | ---: |
| Decoder feature channels / prediction-head hidden channels | 128 / 128 |
| Attention embedding / per-head channels | 128 / 32 |
| Heatmap-decoding candidate pool | 32 |
| Samples per centerline-aligned corridor | 17 |
| Narrow/wide corridor scale | 1.00 / 1.70 |
| Corridor row keep ratio | 0.72 |
| Coarse row keep ratio | 0.78 |
| High-resolution sparse refinement ratio | 0.025 |
| Minimum high-resolution candidates | 96 |
| Predicted structural-scale range before axis/corridor mapping | 0.035--0.24 in normalized horizontal coordinates |
| Mapped centerline-support width (reachable range) | 0.0156--0.042 in normalized horizontal coordinates |
| Mapped corridor-support width | 0.026--0.070 in normalized horizontal coordinates |
| Attention heads | 4 |
| Attention/projection/FFN dropout | 0.10 / 0.10 / 0.10 |
| Narrow/wide corridor-token fusion | 0.75 / 0.25 |
| Longitudinal attention scope | Full row-token sequence with policy attenuation; no fixed local attention window |

### Core operators and implementation details

This subsection makes the implementation behind the paper's structural notation explicit. The equations below describe the released code; lower-level tensor names are linked to the corresponding source locations.

#### Centerline support and coarse row selection

For each coarse feature row, the centerline module normalizes the horizontal center scores and takes their expectation to obtain the center coordinate $\mu_y$. It predicts a row scale and a visibility value $\rho_y$, and renders the spatial support as

$$
S(y,x;\sigma_y)=\rho_y\exp\left[-\frac{1}{2}\left(\frac{x-\mu_y}{\sigma_y}\right)^2\right].
$$

The released profile predicts a raw scale $\omega_y\in[0.035,0.24]$ in the normalized horizontal coordinate system $x\in[-1,1]$ and maps it to two different support widths:

$$
\sigma_y^{\mathrm{center}}
=\operatorname{clip}(0.16\omega_y+0.010,\,0.010,\,0.042),
$$

$$
\sigma_y^{\mathrm{band}}
=\operatorname{clip}(0.26\omega_y+0.016,\,0.026,\,0.070).
$$

Under the released range of $\omega_y$, the reachable centerline-support width is $[0.0156,0.042]$; the value $0.010$ in the first expression is the clipping lower bound. The reachable band width is $[0.026,0.070]$. The first width defines the narrow centerline support, while the second defines the wider structural corridor and the scale used by corridor sampling. Structural injection is continuous through these Gaussian supports; these widths are support scales, not hard horizontal crop boundaries. The implementation is in [`render_axis_gaussian`](models/DecNet.py#L97-L106) and [`ExplicitAxisRepresentation`](models/DecNet.py#L264-L469).

Row selection does **not** use a fixed probability threshold. Coarse rows are ranked by their learned structural scores with a base keep ratio of 0.78. A straight-through top-ranked mask is used in the forward pass, while gradients propagate through the continuous scores. Five-row support dilation and continuity cleaning regularize the selected cranio-caudal span. Consequently, `0.78` is a keep ratio rather than a score threshold. See [`_straight_through_topk_mask`](models/DecNet.py#L120-L141) and the coarse-row construction in [`ExplicitAxisRepresentation`](models/DecNet.py#L331-L372).

#### Visual correlation, corridor sampling, and corridor scoring

At the mid scale, each row is sampled at 17 uniformly spaced offsets $\xi_s\in[-1,1]$. With $\sigma_y^{\mathrm{band}}$ denoting the predicted sampling half-width, the two sets of coordinates are

$$
x_{y,s}^{\mathrm{n}}=\mu_y+\xi_s\sigma_y^{\mathrm{band}},
\qquad
x_{y,s}^{\mathrm{w}}=\mu_y+1.70\,\xi_s\sigma_y^{\mathrm{band}}.
$$

Thus, the narrow corridor has a half-range of $[0.026,0.070]$, and the wide corridor has a half-range of $[0.0442,0.119]$, in the same normalized coordinate system. Off-grid features and structural maps are read with bilinear `grid_sample`, using border padding, coordinate clamping to $[-1,1]$, and `align_corners=True`. The local visual score is produced by a shared pointwise projection, which is the implementation of $w^\top f_{y,s}^q$ in the paper. The structural bias combines the sampled Gaussian band $B_{y,s}^q$ and the aligned coarse sparse prior $P_{y,s}^q$:

$$
b_{y,s}^{\mathrm n}=0.62B_{y,s}^{\mathrm n}+0.38P_{y,s}^{\mathrm n},
\qquad
b_{y,s}^{\mathrm w}=0.66B_{y,s}^{\mathrm w}+0.34P_{y,s}^{\mathrm w}.
$$

The corridor weights and row token are then

$$
\pi_{y,s}^{q}
=\operatorname{softmax}_{s}\!\left(w^\top f_{y,s}^{q}+b_{y,s}^{q}\right),
\qquad
c_y^{q}=\sum_{s=1}^{17}\pi_{y,s}^{q}f_{y,s}^{q}.
$$

The softmax is applied only across the 17 horizontal samples of the same row. Corridor rows are subsequently ranked with a keep ratio of 0.72; this is also a top-ranked policy rather than a fixed score threshold. The implementation is in [`AxisCanonicalStripSampler`](models/DecNet.py#L472-L507) and [`CanonicalStripContext`](models/DecNet.py#L654-L843).

#### Longitudinal propagation

Both coarse row tokens and mid-scale corridor tokens use four-head policy-weighted attention. For a source-row policy $p_j$, the attention multiplier is

$$
g_{ij}=p_j+(1-p_j)\delta_{ij},
$$

and the normalized attention is

$$
\alpha_{ij}^{h}
=\frac{g_{ij}\exp\!\left((Q_i^h)^{\mathsf T}K_j^h/\sqrt{d_h}\right)}
{\sum_{\ell}g_{i\ell}\exp\!\left((Q_i^h)^{\mathsf T}K_{\ell}^h/\sqrt{d_h}\right)}.
$$

Attention and output-projection dropout are both 0.10, and the following token feed-forward block also uses dropout 0.10. The attention is global across the full row-token sequence; the learned policy attenuates rows outside the top-ranked support instead of removing them through a fixed local window. There is therefore no fixed longitudinal neighborhood size. The five-row pooling used around the selection mask regularizes row continuity and must not be interpreted as a five-row attention window. See [`SparsePolicyAttention`](models/DecNet.py#L510-L552), the coarse propagation at [`ExplicitAxisRepresentation`](models/DecNet.py#L369-L372), and the corridor propagation at [`CanonicalStripContext`](models/DecNet.py#L702-L730).

#### Vertebral-chain fusion and high-resolution sparse refinement

The propagated narrow and wide corridor tokens are fused as

$$
\bar c_y=0.75\,\widetilde c_y^{\mathrm n}+0.25\,\widetilde c_y^{\mathrm w}.
$$

Their selection supports are combined separately with narrow/wide weights of 0.60/0.40 and aligned with the coarse sparse prior. The fused row tokens are rendered back to the two-dimensional centerline neighborhood and concatenated with the mid-scale appearance feature and the aligned coarse axis feature. The resulting local fusion mapping consists of a pointwise channel projection followed by two residual depthwise spatial-refinement blocks. Both the token write-back and the residual update are gated by the predicted row and centerline support. The chain residual uses a soft gate with threshold 0.035 and softness 0.018. See [`CanonicalStripContext`](models/DecNet.py#L824-L873).

At high resolution, an appearance score $A$ is support-modulated as $A(0.05+0.95S)$, preserving a 0.05 response floor outside the predicted support, and is then spatially smoothed and ranked. The released profile keeps 2.5% of spatial locations, with at least 96 candidates. The gathered candidates exchange context through four-head attention with 0.10 attention/projection dropout, are scattered back to their original coordinates, and update the high-resolution feature through a support-weighted gate. The final candidate gate uses threshold 0.045 and softness 0.020. See [`LocalCandidateSparseAttention`](models/DecNet.py#L590-L651) and [`PeakOffsetRefiner`](models/DecNet.py#L955-L1032).

Active loss coefficients in the release profile are listed below. Auxiliary entries set to zero in `main.py` are inactive and do not contribute to optimization.

| Loss term in code | Weight |
| --- | ---: |
| `lambda_hm` | 1.00 |
| `lambda_base_hm` | 0.30 |
| `lambda_p2_support_hm` | 0.25 |
| `lambda_p2_hm` | 0.15 |
| `lambda_center_reg` | 1.00 |
| `lambda_corner_reg` | 0.10 |
| `lambda_p2_direct_reg` | 0.25 |
| `lambda_centerline` | 0.40 |
| `lambda_axis_visible` | 0.10 |
| `lambda_row_coverage` | 0.05 |
| `lambda_hm_row_recall` | 0.08 |

These values document the released configuration rather than constituting additional methodological assumptions. The source code remains the authoritative record for lower-level layer composition, interpolation, continuity filtering, and residual mixing.

The implementation groups these terms into three semantic objectives in the paper: vertebral-center detection, vertebral-geometry regression, and spinal-structure learning. Intermediate supervision remains attached to the corresponding semantic objective and does not define an additional prediction task.

## Train, Validation, and Test Separation

The three data roles are explicit and disjoint in the public workflow:

- `--phase train` optimizes the model on `train/` and monitors convergence on `val/`. It never constructs or reads the `test/` dataset.
- `--phase eval` is reserved for final reporting and always reads `test/`.

Validation is used to monitor whether training remains stable. It does not trigger best-epoch checkpoint selection. Training proceeds to the configured final epoch (200 in the reported experiments), `latest_model.pth` stores that final state, and test evaluation is run separately afterward.

## Trained SMSD Checkpoints

Trained SMSD checkpoints are provided through Baidu Netdisk:

- Link: `https://pan.baidu.com/s/1YiB3rLv7D9xU9-d4tkRkAQ?pwd=wxtr`
- Password: `wxtr`

The shared package contains two checkpoint folders:

| Paper setting | Checkpoint folder | Checkpoint path |
| --- | --- | --- |
| `AASCE-98` | `fh_data_bs` | `weights/fh_data_bs/latest_model.pth` |
| `Clinical-150` | `fh_data_lc` | `weights/fh_data_lc/latest_model.pth` |

In other words, the `fh_data_bs` checkpoint corresponds to the `AASCE-98` setting reported in the paper, while the `fh_data_lc` checkpoint corresponds to the `Clinical-150` setting.

After downloading, place the files in the repository as:

```text
weights/
  fh_data_bs/
    latest_model.pth
  fh_data_lc/
    latest_model.pth
```

The HRNet-W32 backbone initializer used for training is separate from these trained SMSD checkpoints. Place `faster_rcnn_hrnetv2p_w32_mstrain_syncbn_1x.pth` directly under `weights/`; the expected file has SHA-256 `33B9A18C6C93AF3ACD92939EFBBA4D705F7B035228BD183A43B9E6208DF2E424`. Its provenance is the Faster R-CNN HRNetV2p-W32, 1x, SyncBN, multi-scale COCO checkpoint from the [official HRNet object-detection model family](https://github.com/HRNet/HRNet-Object-Detection#faster-r-cnn). The loader imports only shape-compatible backbone tensors and reports the matched, missing, ignored, and mismatched keys.

## Repository Structure

```text
datasets/     split-aware dataset loaders and supervision targets
models/       network definition
operation/    training, loss, decoding, and preprocessing utilities
weights/      trained SMSD checkpoints and the HRNet initializer (download separately)
outputs/      prediction and evaluation outputs
check_splits.py  split-count and overlap audit
main.py       entry point for training, validation, inference, and test evaluation
```

## Installation

The reported software environment uses Python 3.10 and CUDA 11.8. A CUDA-enabled PyTorch build compatible with the local CUDA/runtime configuration is required for GPU training. Install the remaining Python dependencies with:

```bash
pip install -r requirements.txt
pip install Pillow
```

## Training with Validation

### Canonical training profile

The settings used by the public training entry point are listed explicitly below.

| Training setting | Released value |
| --- | --- |
| Backbone | HRNet-W32 initialized from the specified pretrained checkpoint |
| Optimizer | AdamW |
| Initial learning rate | $1\times10^{-4}$ |
| Weight decay | $1\times10^{-4}$ |
| Learning-rate schedule | Exponential decay after every epoch |
| Exponential decay factor | 0.98 |
| Training batch size | 1 |
| Validation batch size | 1 |
| Gradient accumulation | 1 step |
| Gradient clipping | L2 norm 5.0 |
| Mixed precision | Enabled when CUDA is available; disabled with `--disable_amp` |
| Training duration | 200 epochs |
| Validation frequency | Every epoch |
| Periodic checkpoint interval | Every 20 epochs |
| Checkpoint used for test reporting | Final epoch-200 checkpoint; validation does not select a best epoch |
| Default command-line seed | 317; independent repetitions should be launched separately with recorded seeds |
| Input size $(H\times W)$: AASCE-128/AASCE-98 | $1280\times512$ |
| Input size $(H\times W)$: Clinical-150 | $1664\times512$ |

The dataset root must contain separate `train/`, `val/`, and `test/` directories. Training accesses only the first two:

```bash
python main.py ^
  --phase train ^
  --data_dir ../data/aasce_98 ^
  --epochs 200 ^
  --batch_size 1 ^
  --val_batch_size 1 ^
  --learning_rate 1e-4 ^
  --weight_decay 1e-4 ^
  --lr_gamma 0.98 ^
  --grad_clip 5.0 ^
  --val_interval 1 ^
  --save_interval 20 ^
  --seed 317 ^
  --use_hrnet_pretrained ^
  --hrnet_pretrained weights/faster_rcnn_hrnetv2p_w32_mstrain_syncbn_1x.pth ^
  --checkpoint_dir weights/aasce_98 ^
  --output_dir outputs/train_aasce_98
```

The reported configuration initializes HRNet-W32 from its pretrained checkpoint, as shown explicitly above. Training from scratch is still possible by omitting both pretrained-weight arguments. The training log records the three semantic objective groups for both training and validation. Checkpoints are saved periodically and at the final epoch; validation does not select a best epoch. For the five-run statistics reported in the paper, each setting is trained five times from independent initial random states, using a separate seed and separate checkpoint/output directories for every run; the epoch-200 checkpoint from each run is evaluated once and the five test results are summarized as mean and standard deviation.

## Inference

### Single-image data path

For each input radiograph, the released model connects its modules in the following order:

1. HRNet produces high-, mid-, coarse-, and semantic-scale feature maps.
2. The semantic feature calibrates the coarse representation, from which the centerline position, structural scale, visibility, and coarse row scores are predicted.
3. The coarse row tokens exchange policy-weighted longitudinal context across the full sequence, while the ranked support controls their structural write-back through the predicted centerline support.
4. The refined centerline defines narrow and wide mid-scale corridors. Appearance and structural scores aggregate each corridor into ordered row tokens, which undergo a second longitudinal propagation to form the vertebral chain.
5. The chain feature and structural support condition the high-resolution candidate scores. Candidate tokens are gathered by the resulting ranking, exchange context, and are scattered back to their original positions before the center heatmap, center offsets, and corner offsets are predicted.
6. Heatmap peaks and their offset fields are decoded into 17 vertebral quadrilaterals, which are restored to original-image coordinates and ordered cranio-caudally.

All centerline, row, corridor, and candidate supports are predicted from the input image. No reference centerline, row mask, vertebral order, or landmark value is used to generate the predictions. The network connection is implemented in [`StructuredSpineDecoder.forward`](models/DecNet.py#L1152-L1324).

Run prediction on a folder of radiographs:

```bash
python main.py ^
  --phase predict ^
  --image_dir ../data/aasce_98/test/images ^
  --checkpoint weights/aasce_98/model_200.pth ^
  --output_dir outputs/predict_aasce_98
```

Main output files:

- `predictions.csv`: predicted vertebral landmark coordinates in original-image space
- `summary.json`: run summary

### Landmark decoding settings

The released decoder applies a $3\times3$ local-maximum suppression operation to the center heatmap, constructs a pool of 32 center candidates, prioritizes candidates with confidence at least 0.05, and returns 17 center responses. The center and ordered corners are recovered as

$$
\widehat P_i^c=(x_i,y_i)+\widehat O^c(y_i,x_i),
\qquad
\widehat P_{i,m}^k=\widehat P_i^c-\widehat O_m^k(y_i,x_i),
$$

where the corner order is top-left, top-right, bottom-left, and bottom-right. The 17 quadrilaterals are sorted by their mean vertical coordinate. In exported `predictions.csv`, zero-based `vertebra_index=0,\ldots,16` therefore corresponds to T1--T12 followed by L1--L5. See [`DecDecoder`](operation/decode.py#L12-L27), [`DecDecoder.ctdet_decode`](operation/decode.py#L157-L245), and the coordinate/order restoration in [`main.py`](main.py#L249-L272).

The standalone `predict` command currently exports the 68 predicted landmark coordinates. The held-out `eval` workflow additionally computes the predicted PT, MT, and TL values and, because reference landmarks are available there, exports their reference counterparts and evaluation errors.

## Evaluation

After training is complete, run final evaluation on the held-out `test/` split:

```bash
python main.py ^
  --phase eval ^
  --data_dir ../data/aasce_98 ^
  --checkpoint weights/aasce_98/model_200.pth ^
  --output_dir outputs/eval_aasce_98_test
```

The `eval` command is restricted to `test/`; validation is performed only inside the training workflow.

Main output files:

- `angles.csv`: predicted and reference Cobb angles
- `predictions.csv`: predicted and reference landmark coordinates
- `point_errors.csv`: vertebral localization errors
- `summary.json`: aggregate evaluation results

### PT, MT, and TL Cobb-angle computation

For vertebra $i$, the evaluation first forms a transverse orientation vector from the midpoints of its left and right edges,

$$
u_i=\frac{P_i^{\mathrm{TR}}+P_i^{\mathrm{BR}}}{2}
-\frac{P_i^{\mathrm{TL}}+P_i^{\mathrm{BL}}}{2}.
$$

The clipped pairwise angle is

$$
\theta_{ij}=\frac{180}{\pi}\arccos\left(
\operatorname{clip}_{[0,1]}
\frac{u_i^{\mathsf T}u_j}{\lVert u_i\rVert_2\lVert u_j\rVert_2+10^{-6}}
\right),
$$

and the cranially ordered dominant end-vertebra pair is $(a,b)=\arg\max_{i<j}\theta_{ij}$. For a one-sided trace, the reported triplet is

$$
(C_{\mathrm{PT}},C_{\mathrm{MT}},C_{\mathrm{TL}})
=(\theta_{1a},\theta_{ab},\theta_{bN}),\qquad N=17.
$$

For a two-sided trace, the evaluation searches the adjacent cranial and caudal subchains:

$$
a_0=\arg\max_{i\leq a}\theta_{ia},\qquad
b_0=\arg\max_{j\geq b}\theta_{bj},\qquad
a_1=\arg\max_{i\leq a_0}\theta_{ia_0}.
$$

If the dominant pair is located in the cranial half of the decoded trace, the triplet is $(\theta_{a_0a},\theta_{ab},\theta_{bb_0})$; otherwise it is $(\theta_{a_1a_0},\theta_{a_0a},\theta_{ab})$. The one-sided/two-sided decision uses the ordered upper- and lower-endplate midpoint trace and a numerical tolerance of $10^{-4}$. The same procedure is applied to predicted and reference landmarks. The returned PT, MT, and TL angles are expressed in degrees and rounded to two decimal places before export. The complete branch logic is implemented in [`cobb_angle_calc`](eval_cobb_calm.py#L171-L337), and the three predicted/reference values are exported by [`run_eval`](main.py#L453-L499).

## Expected Data Layout

### Clinical-150 cohort and acquisition summary

Clinical-150 is the private external test cohort used in the paper. It contains 150 coronal radiographs from 120 patients acquired at one institution between 2019 and 2025. The patient distribution comprises 95 females and 25 males; the median age at the first included examination is 15 years (interquartile range, 13–18; range, 8–73), and 113 of the 150 radiographs were acquired at ages 10–17. Among the 120 patients, 100 contributed one radiograph, 16 contributed two, one contributed three, and three contributed five. The cohort was assembled primarily from adolescents with scoliosis, with other age groups represented; clinically normal examinations were not included. Images with obvious stitching errors or marked blur preventing reliable T1–L5 annotation were excluded during quality control.

Aggregate acquisition information is provided below. Width and height refer to the original image matrix before the model input transformation.

| Acquisition system | Radiographs | Original width × height (pixels) |
| --- | ---: | --- |
| NeuVision New Era | 12 | 2498–3044 × 6364–9161 |
| ANGELL-DR | 68 | 1277–3093 × 4236–7634 |
| GE Optima XR646 | 25 | 1867–2144 × 3936–10640 |
| EOS | 45 | 1749–1956 × 4669–9672 |

Only aggregate cohort and acquisition statistics are documented here; the private clinical images and patient-level records are not distributed with the repository.

Each paper setting uses its own dataset root and its own checkpoints:

```text
data/
  aasce_128/       # 431 train / 50 val / 128 test
    train/{images,labels}/
    val/{images,labels}/
    test/{images,labels}/
  aasce_98/        # 549 train / 60 val / 98 test
    train/{images,labels}/
    val/{images,labels}/
    test/{images,labels}/
  clinical_150/    # 637 train / 70 val / 150 test
    train/{images,labels}/
    val/{images,labels}/
    test/{images,labels}/
```

Before training or final evaluation, the paper-defined counts, label coverage, and pairwise split disjointness can be checked with:

```bash
python check_splits.py --data_dir ../data/aasce_128 --protocol aasce-128
python check_splits.py --data_dir ../data/aasce_98 --protocol aasce-98
python check_splits.py --data_dir ../data/clinical_150 --protocol clinical-150
```

The check fails if the observed counts differ from the selected protocol, if a label is missing, or if any image identifier or image content is shared across train, validation, and test.

Each label file should correspond to one image and store vertebral landmark points in `.mat` format.
