"""Generate synthetic eye-ROI fixtures covering the Phase 3 lighting
condition test matrix.  Run ONCE (idempotent — skips files that exist).

Fixtures are programmatically constructed OpenCV drawings of a stylised
eye (sclera + iris + pupil + eyelid border) with global photometric
adjustments applied to simulate:
  1. normal lighting   (~ mean brightness 110)
  2. bright/overexposed (~ mean brightness 215)
  3. low light / dark  (~ mean brightness 10)
  4. glasses glare     (normal lighting + two saturated specular blobs)

This keeps the repo free of proprietary imagery while still producing
pixels that exercise every step of PreprocessPipeline:
  * cvtColor BGR2GRAY
  * np.mean brightness classification
  * GaussianBlur 5x5
  * threshold dispatch (otsu | adaptive | manual)
  * MORPH_OPEN + MORPH_CLOSE cleanup
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np


FIXTURE_DIR = Path(__file__).resolve().parent


def _draw_eye_base(w: int = 120, h: int = 80) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # --- skin (bright, warm) background across ROI borders ---
    #     Grayscale target: ~205-215 so it's solidly on the "bright" side
    #     of Otsu for the normal fixture.
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (185, 215, 245), thickness=-1)
    # --- sclera ellipse (very bright, almost pure white) ---
    sclera_center = (w // 2, h // 2)
    sclera_axes = (int(w * 0.38), int(h * 0.30))
    cv2.ellipse(canvas, sclera_center, sclera_axes, 0, 0, 360, (245, 250, 255), thickness=-1)
    # --- iris (light hazel-brown, so still brighter than Otsu split) ---
    #     Grayscale target: ~180, on the bright side of Otsu (~160).
    #     Only the TRUE PUPIL is dark, which is exactly what Phase 4 needs.
    cx = w // 2 + 4
    cy = h // 2 + 1
    iris_r = int(min(w, h) * 0.18)
    pupil_r = iris_r // 2
    cv2.circle(canvas, (cx, cy), iris_r, (145, 170, 205), thickness=-1)
    # --- PUPIL — the ONLY truly dark region in the synthetic eye ---
    cv2.circle(canvas, (cx, cy), pupil_r, (8, 8, 8), thickness=-1)
    # --- tiny catchlight on iris (bright) ---
    cv2.circle(canvas, (cx - iris_r // 3, cy - iris_r // 3), 2, (245, 245, 255), thickness=-1)
    # --- eyelid curve (top and bottom, medium tone — still bright side) ---
    #     NOT pitch-black creases; realistic lids are shaded skin not ink.
    for sign, dy in ((1, -int(h * 0.18)), (-1, int(h * 0.18))):
        pts = []
        for t in np.linspace(0.0, 1.0, 40):
            x = int(w * 0.12 + t * (w * 0.76))
            base_cy = h // 2 + dy
            amp = int(h * 0.08)
            y = int(base_cy - sign * amp * math.sin(math.pi * t))
            pts.append((x, y))
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_np], False, (130, 160, 195), thickness=2, lineType=cv2.LINE_AA)
    # --- Eyelashes: sparse, medium-dark pixels at the lid margin only. ---
    #     Kept intentionally mild: too many dark specks → Otsu merges them
    #     with the pupil and blows the fg ratio.  These are tiny single-pixel
    #     dots at the upper/lower lid edges only.
    rng = np.random.default_rng(42)
    lash_count = 0
    for _ in range(200):
        ex = rng.integers(int(w * 0.12), int(w * 0.88))
        ey = rng.integers(0, h)
        # only along upper and lower lid margins (3px bands)
        upper_lid_y = h // 2 - int(h * 0.18) - 2
        lower_lid_y = h // 2 + int(h * 0.18) + 2
        if not (abs(ey - upper_lid_y) <= 3 or abs(ey - lower_lid_y) <= 3):
            continue
        # and never inside the iris area
        if math.hypot(ex - cx, ey - cy) < iris_r * 1.1:
            continue
        if lash_count >= 25:
            break
        cv2.circle(canvas, (ex, ey), 0, (90, 110, 145), thickness=1)
        lash_count += 1
    return canvas


def _brightness_mean(img: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))


def _save(name: str, img: np.ndarray) -> None:
    out = FIXTURE_DIR / name
    if out.exists():
        return
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[fixtures] wrote {out}  mean_gray={_brightness_mean(img):.1f}")


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    w, h = 120, 80  # must match defaults of _draw_eye_base()
    base = _draw_eye_base(w, h)

    # ------------------------------------------------------------------
    # (1) Normal lighting — target mean_gray ~112 (matches test band
    #     80-160).  The histogram must stay BIMODAL so Otsu isolates
    #     the pupil blob alone (fg ratio 0.5–25%).
    # ------------------------------------------------------------------
    # Uniform dimming so the bright peaks (sclera ~130, skin ~112) are
    # solidly separated from the dark pupil peak (~4) in grayscale space.
    normal = cv2.convertScaleAbs(base, alpha=0.52, beta=0)
    rows_n, cols_n = normal.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols_n, cols_n / 2.5)
    kernel_y = cv2.getGaussianKernel(rows_n, rows_n / 2.5)
    vignette = (kernel_y @ kernel_x.T)
    vignette = vignette / vignette.max()
    vignette = 0.92 + 0.08 * vignette  # very subtle corner darkening only
    for c in range(3):
        normal[:, :, c] = np.clip(normal[:, :, c] * vignette, 0, 255).astype(np.uint8)
    _save("normal_lighting.jpg", normal)

    # ------------------------------------------------------------------
    # (2) Bright / overexposed — mean gray ~231 (still ≤ 240 gate so
    #     lighting_ok=True with default config).
    # ------------------------------------------------------------------
    bright = cv2.convertScaleAbs(base, alpha=0.98, beta=40)
    bright = np.clip(bright.astype(np.int16), 0, 248).astype(np.uint8)
    _save("bright_overexposed.jpg", bright)

    # ------------------------------------------------------------------
    # (3) Low light — hard constraints:
    #       (a) mean_gray < 15  →  lighting_ok=False gate fires
    #       (b) large ENOUGH local pupil-vs-sclera contrast that after
    #           5×5 Gaussian blur + 25×25 adaptive neighbourhood, the
    #           pupil (now clamped to 0–2) is still clearly below the
    #           adaptive threshold while sclera/iris (~17) stays above.
    # ------------------------------------------------------------------
    # Step 1: build a pupil-vs-rest MASK so contrast can be injected
    #         precisely into each zone without affecting the other.
    cx_l, cy_l = w // 2 + 4, h // 2 + 1
    iris_r_l = int(min(w, h) * 0.18)
    pupil_r_l = iris_r_l // 2  # = 7
    yy_m, xx_m = np.ogrid[:h, :w]
    d2 = (xx_m - cx_l) ** 2 + (yy_m - cy_l) ** 2
    pupil_only = (d2 <= pupil_r_l ** 2).astype(np.float32)

    # Step 2: crush everything, then scale bright-zone and dark-zone
    #         separately to create contrast WHILE keeping global mean < 15.
    low_pre = cv2.convertScaleAbs(base, alpha=0.05, beta=0)  # everything ~0-12
    low_f = low_pre.astype(np.float32)
    # Darken pupil explicitly to near-zero to survive GaussianBlur:
    low_f = low_f * (1.0 - pupil_only[:, :, None]) * 0.95  # rest stays dim
    low_f = np.clip(low_f, 0, 11).astype(np.uint8)
    # Step 3: lift sclera + iris ring up to ~16 gray (BUT NOT pupil).
    #         We use the ellipse mask MINUS the pupil disk to only
    #         brighten the non-pupil eye area.
    dist_e = np.sqrt(((xx_m - cx_l) / (w * 0.36)) ** 2 + ((yy_m - cy_l) / (h * 0.28)) ** 2)
    eye_zone = (dist_e <= 1.0).astype(np.float32)
    eye_not_pupil = np.clip(eye_zone - pupil_only, 0.0, 1.0)
    low_f2 = low_f.astype(np.float32)
    low_f2 += eye_not_pupil[:, :, None] * 11.0  # sclera → ~0+11 = 11, iris+pre_bright → ~3+11 = 14
    # Also bump the surrounding skin slightly (from ~0-2 to ~3-5) so it
    # doesn't also count as "pupil-dark" to the adaptive windows near
    # the image borders.
    outer_zone = (dist_e > 1.0).astype(np.float32)
    low_f2 += outer_zone[:, :, None] * 3.0
    low = np.clip(low_f2, 0, 18).astype(np.uint8)
    # Step 4: tiny read noise, kept to ±1 to avoid erasing the delta.
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 2, size=low.shape, dtype=np.uint8)
    low = cv2.add(low, noise)
    # Final clamp caps the bright end so mean can't drift over 15.
    low = np.clip(low.astype(np.int16), 0, 19).astype(np.uint8)
    _save("low_light.jpg", low)

    # (4) Glasses glare — start from NORMAL (not bright) then add two
    #     moderate specular reflection blobs on the lens area.
    #     Mean stays inside the valid lighting band.
    glare = normal.copy()
    for (gx, gy, rad, inten) in (
        (int(0.22 * 120), int(0.35 * 80), 8, (230, 240, 250)),
        (int(0.78 * 120), int(0.30 * 80), 6, (225, 235, 250)),
    ):
        cv2.circle(glare, (gx, gy), rad, inten, thickness=-1)
    _save("glasses_glare.jpg", glare)

    # Report
    print()
    print("fixture summary:")
    for name in (
        "normal_lighting.jpg",
        "bright_overexposed.jpg",
        "low_light.jpg",
        "glasses_glare.jpg",
    ):
        p = FIXTURE_DIR / name
        if p.exists():
            img = cv2.imread(str(p))
            print(f"  - {name:30s}  shape={img.shape}  mean_gray={_brightness_mean(img):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
