# PACE — Phase 3 Eye-ROI Fixtures

This directory holds 120×80 px eye-ROI crops consumed by
`tests/test_preprocess.py`.  Each file covers one lighting condition in
the project's evaluation test matrix:

| File | Lighting condition | Mean gray | Source |
|---|---|---|---|
| `normal_lighting.jpg` | Frontal eye, diffuse indoor lighting (≈ 300 lux) | ~112 | Synthetic, deterministic OpenCV draw |
| `bright_overexposed.jpg` | Outdoor / strong ring-light, crushed shadows above 80 | ~231 | Synthetic, photometric variant of normal |
| `low_light.jpg` | Dim webcam in nocturnal room (< 50 lux, noisy) | ~12 | Synthetic, darken + Poisson-style noise |
| `glasses_glare.jpg` | Normal lighting + two lens specular reflections | ~116 | Synthetic, additive blobs on normal |

## Regenerating the fixtures

The generator is idempotent — it skips files that already exist.  Force
a rebuild by deleting the JPEGs first:

```bash
cd <repo-root>
rm tests/fixtures/eye_rois/*.jpg
python tests/fixtures/eye_rois/_generate.py
```

## Why synthetic fixtures?

Per the hackathon project charter, the public repo must not contain
third-party imagery with ambiguous licensing or PII (faces of non-consenting
individuals).  The programmatic fixtures exercise *every* OpenCV code path
in `app/detection/preprocess.py` — BGR2GRAY conversion,
`np.mean(gray)` lighting classification, Gaussian blur, all three
threshold dispatch modes, and morphological open/close cleanup — while
remaining reproducible across CI environments.

When replacing with real captured ROIs later (e.g., from
`python -m app.main --detection-test` output), simply overwrite the
files above keeping the names identical — the test suite will pick up
the real imagery automatically without any code change.
