# PACE Gaze Feature: Pupil-Radius Normalization for Head-Movement Robustness

## 1. The Problem PACE Solves

Most webcam-based eye trackers fail the moment the user leans in, leans back, or shifts their head even a few centimetres sideways. Raw pixel-based pupil coordinates scale with camera distance, so identical gaze directions produce wildly different numeric values depending on how far the face is from the lens.

Without handling this, the 9-point calibration (Phase 7) would need to be re-run every time the user slumps forward in their chair — a showstopper for any "hands-free" claim.

## 2. The Core Insight

The corneal reflection (glint) is a virtual image of the light source formed by the cornea's front surface. As the eye rotates, the pupil's centre moves relative to this fixed reflection anchor — the vector from glint to pupil centre encodes *gaze direction* independently of eyeball position.

Crucially, when camera distance changes, **both the pupil–glint pixel distance AND the pupil's pixel radius scale by the same proportion**. Dividing one by the other cancels the scale factor:

```
Close to camera:   |pupil−glint| = 20 px,   pupil radius = 20 px   →  20/20 = 1.0 pupil-radius units
Far from camera:   |pupil−glint| = 10 px,   pupil radius = 10 px   →  10/10 = 1.0 pupil-radius units
```

Same gaze direction, different head distance → identical normalized feature. That is the entire trick.

## 3. Step-by-Step Pipeline

### 3.1 Per-Eye Raw Vector (Phase 6 `compute_eye_vector`)

Inputs per eye: `PupilResult` (centre `(cx_p, cy_p)`, radius `r`), `GlintResult` (centre `(cx_g, cy_g)`):

```
raw_dx = cx_p - cx_g
raw_dy = cy_p - cy_g
scale  = max(r, ε)          # ε = 1e-3 guards against degenerate r=0 detections
norm_dx = raw_dx / scale
norm_dy = raw_dy / scale
```

Units are now "pupil-radii" rather than ROI pixels. The sign convention matches image-space coordinates: `+dy = pupil below glint = looking down on screen`.

Per-eye confidence is the **geometric mean** of input confidences rather than the arithmetic mean:

```
vector_confidence = √(pupil.confidence · glint.confidence)
```

This choice penalises a vector if *either* input is weak. If pupil = 0.8 but glint = 0.2: arithmetic mean = 0.50 (looks acceptable), geometric mean = 0.40 (correctly flagged as below the 0.4 gate when `min_*_confidence = 0.4`). A single-bad-input penalty is what we want, because a vector built on a unreliable glint or a partially-occluded pupil would inject noise into calibration.

### 3.2 Confidence Gating

Before any math, `found=False` or confidence below `min_pupil_confidence` / `min_glint_confidence` (both default 0.4) returns `None`. Fabricating a vector from unreliable inputs does not help; the fusion step (§3.3) gracefully degrades to single-eye mode instead.

### 3.3 Two-Eye Fusion (`fuse`)

Three modes, configurable at runtime:

| Mode | Behaviour | Use case |
|------|-----------|----------|
| `confidence_weighted` (default) | `fused = (L·w_L + R·w_R) / (w_L + w_R)` where `w = vector_confidence` | Production. A half-occluded right eye (low w) contributes proportionally less without being discarded entirely. |
| `average` | `fused = (L + R) / 2` | Baseline evaluation: "does the weighted variant actually beat a simple mean?" |
| `best_eye` | `fused = higher_confidence(L, R)` (ties → left) | Debugging when you suspect the weighting logic itself is misbehaving. |

Fused confidence = `max(left.conf, right.conf)`. We use max rather than mean because "at least one eye is solid" is the meaningful lower bound for downstream calibration; a mean would artificially punish a system where only one eye had its eyelid momentarily occluded.

Single-eye fallback is the intentional graceful degradation: if one eye is fully missing (e.g. user turns their head 45° so only one eye is visible), `fuse` returns the surviving eye's vector verbatim with `eyes_used` set to `"left"` or `"right"`. Phase 12's evaluation module splits accuracy metrics by `eyes_used` to quantify this degradation numerically.

### 3.4 Lightweight Smoothing (rolling window)

Before returning to callers, Phase 6 applies its own `smoothing_window`-frame simple moving average (default 3 frames) via an internal `collections.deque(maxlen=smoothing_window)`.

**Why this is separate from Phase 9's One-Euro filter:**
- One-Euro is a *cursor-level* filter: it trades latency for smoothness at the final output, tuned on the assumption the gaze→screen mapping is already good.
- The rolling window at this stage is a *feature stabilizer*: the 9-point calibration procedure (Phase 7) averages several readings per calibration point. If the raw feature has single-frame noise spikes, calibration itself learns those spikes as part of the mapping, producing a jittery screen cursor no amount of downstream filtering can fix.

Call `GazeFeatureExtractor.reset()` after recalibration or pause/resume cycles to clear the smoothing history — same API pattern as Phase 4's `PupilDetector.reset()`.

## 4. Contract for Phase 7 (Calibration) and Phase 8 (Mapping)

The only fields those phases read from `GazeFeature`:

```python
gf.vector_x    # float, pupil-radius units, head-distance-invariant
gf.vector_y    # float, pupil-radius units, head-distance-invariant
gf.confidence  # float, ∈ [0, 1]
```

Everything else (`timestamp`, `fusion_mode_used`, `eyes_used`) is for Phase 12 metrics and Phase 13 debug overlays. Calibration and mapping should **not** branch on `eyes_used` or `fusion_mode_used`; they should treat the feature as a black-box 2D point-with-confidence.

## 5. Empirical Proof (Informal Judge Demo)

Run:

```
python -m app.main --feature-test
```

The console prints one line per second for 8 seconds:

- `t=0…3s` – hold your gaze on one fixed spot (e.g. the corner of your monitor).
- `t=3…8s` – still looking at the SAME spot, gently sway your head left, right, forward (close to webcam), and backward.

Observe:
- Raw pixel column `[raw L pupil=(XX.X, YY.X) r=...]` drifts noticeably (forward motion increases r by 20-50%; sideways motion shifts the ROI's internal origin).
- Normalized `vector_x` / `vector_y` columns stay within a ±0.1 pupil-radius band around the same steady value they had during t=0…3s.

That side-by-side contrast — raw drifting vs normalized stable — is the informal head-movement-robustness proof Phase 12's formal benchmark will later quantify with a proper motion rig or pandas-based trajectory analysis.

## 6. Edge Cases & Why They Work

| Case | Mechanism |
|------|-----------|
| `PupilResult.radius = 0.0` (degenerate detection) | `max(r, 1e-3)` prevents division by zero; returned `scale_used=1e-3` is visible in debug output so Phase 13 GUI can flag the frame as suspect without crashing. |
| Glint sits *on* pupil centre (`dx=0, dy=0`) | Result is `(0, 0)` — mathematically valid. Geometrically this means the light source is aligned along the eye's optical axis; gaze is straight-ahead. The 9-point calibration will learn the (0,0) → screen-centre mapping normally. |
| One eye fully occluded | Per-eye returns `None` → fusion returns the other eye's vector with `eyes_used` set. Feature still valid. |
| Both eyes low-confidence → both `None` | `compute()` returns `None`. Phase 7/8 should treat this as "no gaze signal this frame" (hold cursor, increment a consecutive-misses counter for Phase 14 user-facing "lost tracking" state). |
| `eye_scale_source = "fixed"` config fallback | Uses a constant 15 px instead of the detected radius. Useful only for debugging "is the normalization step causing weirdness?" by temporarily disabling it; NOT recommended for production since you lose the head-robustness property. |
