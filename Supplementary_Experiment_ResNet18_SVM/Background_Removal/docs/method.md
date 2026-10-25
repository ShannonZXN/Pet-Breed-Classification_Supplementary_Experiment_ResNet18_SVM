# Processing method

[← Back to README](../README.md)

## Inputs and coordinate alignment

Each original is a JPG/JPEG image paired with a PNG trimap with the same complete
filename stem. The original is decoded to RGB. The mask is read as stored pixels,
without conversion to RGB or grayscale: converting a palette PNG could replace
its useful label indices with palette colors.

A mask must be two-dimensional, have NumPy dtype `uint8`, have the same height
and width as its original, and contain only labels 1, 2, and 3. Not every label
must occur in every mask. All matching is case-insensitive and nonrecursive.

The code does not apply EXIF orientation, resizing, cropping, padding, or
normalization. Mask coordinates therefore stay aligned with the stored image
coordinates. Comparison sheets add labels around panels, but the separately
saved image outputs contain no captions or added borders.

## Keep mask and erosion

Let `L` denote the trimap and let `K0` be the initial binary keep mask:

```python
K0 = (L == 1) | (L == 3)
```

Label 2 is excluded immediately. For radius `r > 0`, the structuring element
includes integer offsets satisfying `dx**2 + dy**2 <= r**2`, within a square
of side length `2*r + 1`. The same integer radius is applied to every image;
it is measured in original-image pixels rather than as a fraction of image size.

One SciPy binary erosion is applied with `iterations=1` and `border_value=1`.
This boundary value treats positions outside the photograph as retained rather
than introducing artificial background along the photograph's edges.

When animal protection is enabled:

```python
K = eroded_mask | (L == 1)
```

When protection is disabled, `K` is the eroded mask directly. For `r = 0`, the
operation returns `K0` without erosion in either mode. No label-2 pixel is ever
included, and protection restores only label 1.

## RGB output

The output array has the same dimensions and dtype as the decoded RGB image:

```python
result = np.zeros_like(original)
result[K] = original[K]
```

Retained RGB values are copied exactly from the decoded original. Removed
pixels become `[0, 0, 0]`. The result is an opaque RGB PNG with no alpha channel.
This statement concerns decoded pixels; it does not claim equality with the
original JPEG's compressed bytes or image metadata.

## What the reports establish

- The processing log records the initial/final keep counts and removed label-1
  and label-3 pixels. Current scripts also record the protection setting per row.
- The run summary's `mask_mapping` describes the initial label mapping. Its
  `erosion` object describes the subsequent erosion and protection operation.
- Batch filename audits are written before image processing begins. Final
  processing logs and summaries are written after processing finishes.
- The separate audit tool checks filename presence and last-run membership,
  not pixel correctness, image readability, or segmentation quality.

## Practical limits

Protection guarantees preservation of pixels *labeled* as animal. Real fur or
other animal detail placed in label 3 can still be eroded. Some label-3 pixels
can remain after erosion, so the output is not guaranteed to exclude every
background-colored pixel inside the original uncertain boundary.

A fixed pixel radius has different relative effects on differently sized
images. Inspect sample comparisons before choosing a radius, and keep that
choice consistent with the reported experiment. Background replacement may
affect a downstream model's input distribution; this utility alone does not
establish whether classification accuracy improves.

The masks are supplied annotations, so this workflow depends on their
availability. It does not provide segmentation for new, unannotated images.
