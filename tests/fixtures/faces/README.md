# Face / Eye Detection Test Fixtures

This directory contains fixture images used by `tests/test_face_eye_detector.py`.

## Files currently present (synthetic / generated)

| File | Source | Purpose |
|------|--------|---------|
| `no_face_empty_room.jpg` | Programmatically generated (solid gradient + Gaussian noise) in `conftest.py` helper | Guaranteed "no face" negative sample — exercises the real detection pipeline's not-found path and miss-counter logic. |

## Files you can drop in (OPTIONAL — for integration testing)

If you place any of the following **royalty-free or self-captured** JPEGs here, the test
suite will prefer them over the mock path for the corresponding assertions:

| Expected filename | Suggested subject |
|-------------------|--------------------|
| `normal_frontal.jpg`  | Adult, normal indoor lighting, looking straight at the camera (no glasses). ~640x480 or larger. |
| `glasses.jpg`         | Same person wearing clear eyeglasses, similar lighting/pose. |
| `slight_angle.jpg`    | Head turned ~15-25 degrees away from frontal. |
| `no_face_empty_room.jpg` | Any frame with no human faces (wall, desk, outdoor scene). |

The test suite auto-detects their presence via `pathlib.Path.exists()`, so no code
changes are needed.

## Recommended sources for fixture photos

All of these provide CC0 / public-domain images suitable for a hackathon repo:

- **Unsplash** (https://unsplash.com) — search "face portrait indoor", "man with glasses", "empty office room".
- **Pexels** (https://www.pexels.com) — same search terms.
- **Self-captured** — a quick 640x480 webcam selfie works perfectly; the evaluation target is ≥95% detection on the normal-lighting frontal set.
