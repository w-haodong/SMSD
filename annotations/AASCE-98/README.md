# AASCE-98 landmark annotations

This release provides 98 study-curated landmark annotation files corresponding to the original AASCE 2019 challenge test images. These are annotations provided by this study, not the challenge organizers' reference annotations. Radiographs and visualization images are not included.

## Format and correspondence

- `labels/01-July-2019-1.jpg.mat` corresponds to the original image `01-July-2019-1.jpg`; the same naming rule applies to image IDs 1–98.
- Each MAT file contains only `p2`, a **68 × 2** array of `(x, y)` coordinates.
- Coordinates use **zero-based pixels in the original, uncropped image**, not resized network-input coordinates. No offset should be added when using the Python evaluation pipeline.
- Every consecutive four points describe one vertebra in the order **top-left, top-right, bottom-left, bottom-right**.
- The 17 vertebral groups are ordered from **T1–T12, then L1–L5**.
- If an image is resized or cropped, apply the same spatial transformation to its coordinates before evaluation.

```python
from scipy.io import loadmat

points = loadmat("labels/01-July-2019-1.jpg.mat")["p2"]
assert points.shape == (68, 2)
```

## Annotation version

The released coordinates are identical to the user-confirmed annotation snapshot consolidated on **2026-09-08**. It combines 84 previously matched FH annotations, eight model-assisted annotations accepted after user review, and six manually corrected annotations. Four of the manually corrected cases had their vertebral groups reordered after editing; the final files already contain the corrected group order.

All 98 files were checked for image correspondence, finite coordinates, original-image bounds, and top-to-bottom vertebral group order. The public files omit local paths, intermediate predictions, confidence values, and checkpoint metadata without changing the final `p2` values or their order.
