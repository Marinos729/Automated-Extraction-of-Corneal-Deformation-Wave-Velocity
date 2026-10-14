# Corneal Deformation-Wave Velocity

Automated extraction of the **propagation velocity of the corneal deformation feature**
from Corvis ST high-speed Scheimpflug sequences.

This is the code accompanying the MSc thesis *"In the Storm of the Eye: Automated
Extraction of Corneal Deformation Wave Velocity"* (TU Delft, Biomedical
Engineering). The Corvis ST records the dynamic deformation of the cornea under a
standardised air puff, but reduces each sequence to a handful of scalar indices
measured at the corneal apex. This pipeline instead tracks the bright deformation
feature as it travels across the corneal cross-section and reports its lateral
propagation velocity, a spatially resolved descriptor that the standard indices
do not provide.

> **What the velocity is.** The extracted quantity is the velocity of the
> macroscopic deformation *front* (on the order of 0.1 m/s), not an elastic-wave
> speed. It is one to two orders of magnitude slower than the guided elastic
> waves measured by optical coherence elastography and cannot be converted to a
> shear modulus. See the thesis Discussion for the full argument.

---

## Repository contents

| File | Purpose |
|------|---------|
| `wave_pipeline_unified.m` | The complete pipeline. Reads a dataset tree (or a single subject), segments the cornea, builds an arc-length geometry, and extracts the velocity with the intensity-profile method, cross-checked by an independent difference-based method. The same file also runs the synthetic ground-truth verification through its `verify` mode. |
| `stats_tests.m` | Inferential statistics on the pipeline output (`all_results.csv`): age comparison (two-sample *t*-test with an F-test for equal variance), meridian comparison (Wilcoxon signed-rank with a Hodges-Lehmann interval), and the supporting Shapiro-Wilk normality checks. |

---

## Requirements

- **MATLAB** (a recent release; no version-specific language features are used).
- **Image Processing Toolbox** — required by `wave_pipeline_unified.m`
  (`edge`, `medfilt2`, `bwconncomp`, `regionprops`, `imresize`).
- **Statistics and Machine Learning Toolbox** — required by `stats_tests.m`
  (`ttest2`, `signrank`, `vartest2`, `norminv`, `normcdf`). Its vector-figure
  output uses `exportgraphics` (R2020a or newer); a `print` fallback is noted in
  the file. The Shapiro-Wilk test is implemented in the file itself, so no extra
  download is needed.

No external dependencies beyond MATLAB and these toolboxes.

---

## Expected data layout

The pipeline infers the age group, eye, and subject from the last three folder
levels, so the raw frames must be arranged as:

```
<rawRoot>/<AgeGroup>/<EyeSide>/<Subject>/*.JPG
```

Both image sides (R and L) of each subject are processed. The left image side is
mirrored to a common right-facing orientation, and the image side is translated
into an anatomical meridian (nasal / temporal) internally.

---

## Usage

### 1. Extract velocities from the dataset

```matlab
% whole dataset tree, saving per-sequence diagnostic figures
wave_pipeline_unified('tree', rawRoot, outRoot);

% numbers only, no figures (faster)
wave_pipeline_unified('tree', rawRoot, outRoot, false);

% a single subject folder
wave_pipeline_unified('one', subjectPath, outRoot);
```

`saveImages` (4th argument, default `true`) toggles every PNG. Set it to `false`
for a fast run that still writes the results table.

### 2. Run the synthetic verification

```matlab
wave_pipeline_unified('verify', outRoot);
```

### 3. Run the statistics

```matlab
stats_tests('all_results.csv');
```

---

## Outputs

### Main pipeline (`wave_pipeline_unified.m`)

Written under `outRoot`:

- `all_results.csv` — one row per processed sequence (schema below).
- `all_tracked_results.mat` — the raw binary tracker outputs.
- `master_summary.png` — overview of the run (only when `saveImages` is true).
- `<AgeGroup>/<EyeSide>/<Subject>_<imgSide>_<meridian>/` — one folder per
  sequence, each containing `result.mat` (per-frame tracks and the fit) and,
  when `saveImages` is true, `fusion_summary.png` (the profile track with its
  fit, the binary track overlaid for comparison, the residuals, and the grade).

**`all_results.csv` columns**

| Column | Meaning |
|--------|---------|
| `age`, `eye`, `subject`, `img_side`, `meridian` | Identifiers for the sequence. |
| `n_frames` | Frames in the analysis window. |
| `n_profile`, `n_binary`, `n_both` | Frames localised by the profile method, the binary method, and both. |
| `n_agree`, `n_conflict`, `n_interp` | Co-detected frames that agreed, conflicted, or were filled by interpolation. |
| `agree_rate` | `n_agree / n_both` (the verification agreement between the two methods). |
| `delta_px` | Constant leading-edge offset between the two methods (arc px). |
| `span_start`, `span_end` | First and last frame the feature was tracked over. |
| `v_profile` | **Profile velocity (arc px/frame). This is the reported velocity.** |
| `v_profile_m_s` | The same velocity in m/s (`v_profile` times the calibration constant). |
| `v_binary` | Binary-method velocity (arc px/frame), used only to verify `v_profile`. |
| `R2` | Straight-line fit quality of the profile trajectory. |
| `phi_deg` | Fitted obliquity angle of the sampling bands. |
| `grade` | A–F quality grade for the sequence (based on the profile track and the two-method agreement). |
| `offset_warning` | Flag for a suspected one-frame offset between methods. |

To convert any arc-px/frame velocity to metres per second, multiply by the device
calibration (below).

### Verification (`wave_pipeline_unified('verify', outRoot)`)

Written under `outRoot/verification/`: `verification_log.txt`,
`fig_synthetic_data.png`, `fig_recovery_velocity.png` with
`verification_velocity.csv`, `fig_recovery_snr.png` with `verification_snr.csv`,
and the arc-length-bias (`curvature`) figure and CSV. The verification reports
the recovery of the **profile** velocity, the same quantity used on the real data.

### Statistics (`stats_tests.m`)

Prints the test statistics (age *t*-test, F-test for equal variance,
Shapiro-Wilk normality for each age group and for the within-eye differences,
and the meridian Wilcoxon signed-rank with its Hodges-Lehmann interval), and
writes `stats_results.csv`, `fig_normality.png` (histogram and normal Q-Q plot
of the within-eye differences), and `velocity_age.pdf` and
`velocity_meridian.pdf` (box-and-point plots by group).

---

## Calibration

Velocities are converted from native arc pixels per frame to physical units with

```
v[m/s] = (mmPerPx / msPerFrame) * v[arc px/frame] = 0.063898 * v[arc px/frame]
```

where `mmPerPx = 8.5 / 576` mm (8.5 mm across 576 pixels) and
`msPerFrame = 1000 / 4330` ms (a 4330 frames/s acquisition). Both constants live
in `defaultParams` inside the pipeline.

---

## Method, in brief

1. Each frame is cropped to the corneal region of interest and the left image
   side is mirrored to a common orientation.
2. The anterior and posterior corneal surfaces are segmented and temporally
   regularised, and a single tissue-following arc-length geometry is built per
   frame.
3. The **intensity-profile method** (primary) samples depth-averaged profiles
   along the arc at a fitted obliquity angle, localises the bright feature in
   each frame, and fits a straight line to the positions. Its slope is the
   reported velocity.
4. An independent **difference-based method** (verifier) warps each frame,
   differences consecutive frames, and tracks the resulting blob. It is used only
   to confirm that both methods follow the same feature, not to produce the
   reported number.

The numeric path is deterministic (exhaustive RANSAC, no random sampling);
`rng(seed)` only fixes the jitter of diagnostic scatter plots.

---

## Notes and known limitations

- **Profile velocity is the single reported quantity.** The pipeline reports the
  slope of the profile track (`v_profile`, and `v_profile_m_s`). The binary
  method is an independent verifier only: it contributes the agreement statistics
  (`n_agree`, `agree_rate`, `delta_px`) that confirm the two methods followed the
  same feature, and it is overlaid in the diagnostic figure, but it does not set
  the velocity. There is no fused velocity.
- **Verified, not validated.** The pipeline is checked against a synthetic ground
  truth (the `verify` mode) and by the agreement of its two methods, but it has
  not been validated against a manually labelled ground truth on real images.
  Image noise is high and the feature is not clearly distinguishable in every
  sequence.

---

## Citation

If you use this code, please cite the thesis:

> Marinos, D. *In the Storm of the Eye: Automated Extraction of Corneal
> Deformation Wave Velocity.* MSc thesis, Delft University of Technology,
> Biomedical Engineering, 2026.

---

## License

No license is set yet. Until one is added, the default is "all rights reserved":
others may view the code but have no granted rights to reuse it.
