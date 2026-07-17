"""
Handcrafted per-image feature vector for the STATISTICAL domain-gap method.

Compact, interpretable descriptor over color / photometric / texture /
frequency cues, computed over the valid pixels (whole crop for scene/bbox,
fragment-only for the masked level). The vector feeds MMD / energy distance /
proxy-A-distance; individual channels (e.g. hue) feed the hue-EMD companion.
"""
import cv2
import numpy as np

FEATURE_NAMES = [
    "H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std",
    "L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std",
    "gray_mean", "gray_std",
    "edge_density", "grad_mean", "grad_std", "lap_var",
    "hf_energy_ratio", "entropy",
]


def _stats(vals):
    return float(vals.mean()), float(vals.std())


def handcrafted(img_rgb, mask=None):
    """Return (feature_vector[20], hue_values[valid])."""
    img = img_rgb.astype(np.uint8)
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)
    if mask.sum() < 16:  # degenerate; treat whole crop as valid
        mask = np.ones(img.shape[:2], dtype=bool)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] *= 2.0  # OpenCV hue is 0..179 -> degrees
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    m = mask
    feats = []
    for ch in range(3):
        feats += list(_stats(hsv[..., ch][m]))
    for ch in range(3):
        feats += list(_stats(lab[..., ch][m]))
    feats += list(_stats(gray[m].astype(np.float32)))

    # texture / edges over valid region
    edges = cv2.Canny(gray, 60, 160) > 0
    edge_density = float(edges[m].mean())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean, grad_std = _stats(grad[m])
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(lap[m].var())

    # frequency: fraction of spectral energy above mid-band (full crop)
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray.astype(np.float32))))
    cy, cx = np.array(f.shape) // 2
    yy, xx = np.ogrid[:f.shape[0], :f.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    hf = f[r > 0.25 * r.max()].sum()
    hf_ratio = float(hf / (f.sum() + 1e-8))

    # intensity entropy over valid region
    hist = np.bincount(gray[m].astype(np.int64), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    feats += [edge_density, grad_mean, grad_std, lap_var, hf_ratio, entropy]
    return np.asarray(feats, dtype=np.float32), hsv[..., 0][m]


# ---------------------------------------------------------------------------
# Extended descriptor: base + canonical texture features (GLCM/Haralick, LBP,
# Gabor). Used only as a robustness check on the compact descriptor above.
# ---------------------------------------------------------------------------
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy",
              "correlation", "ASM"]
_LBP_P, _LBP_R, _LBP_BINS = 8, 1, 10       # uniform LBP -> P+2 = 10 bins

EXT_FEATURE_NAMES = (
    FEATURE_NAMES
    + [f"glcm_{p}" for p in GLCM_PROPS]
    + [f"lbp_{i}" for i in range(_LBP_BINS)]
)


def handcrafted_ext(img_rgb, mask=None):
    """
    Return (extended_feature_vector, hue_values).

    Adds Haralick/GLCM and uniform LBP to `handcrafted`.
    LBP is pooled over the valid pixels only; GLCM needs a
    rectangular window and is therefore computed over the whole crop (with the
    belt zeroed at the mask level), which is applied identically to every
    domain and so does not favour one of them.
    """
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    base, hue = handcrafted(img_rgb, mask)
    img = img_rgb.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if mask is None or mask.sum() < 16:
        mask = np.ones(gray.shape, dtype=bool)

    # --- GLCM / Haralick (averaged over 4 angles, 2 distances) ---
    q = (gray // 4).astype(np.uint8)            # 64 grey levels
    glcm = graycomatrix(q, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    glcm_feats = [float(graycoprops(glcm, p).mean()) for p in GLCM_PROPS]

    # --- uniform LBP histogram over valid pixels ---
    lbp = local_binary_pattern(gray, _LBP_P, _LBP_R, method="uniform")
    hist, _ = np.histogram(lbp[mask], bins=_LBP_BINS, range=(0, _LBP_BINS))
    hist = hist.astype(np.float64)
    lbp_feats = list(hist / (hist.sum() + 1e-12))

    return (np.asarray(list(base) + glcm_feats + lbp_feats,
                       dtype=np.float32), hue)
