import os
import numpy as np
import cv2

# ======================
# Paths
# ======================
CALIB_PATH  = "camera_calib.npz"
MATCHES_PATH = "outputs/matches_inliers_px.npz"
IMG1_PATH = "object/img1.png"
IMG2_PATH = "object/img2.png"

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# Load
# ======================
cal = np.load(CALIB_PATH)
K = cal["K"].astype(np.float64)
dist = cal["dist"].astype(np.float64)

m = np.load(MATCHES_PATH)
pts1_px = m["pts1"].astype(np.float64)  # Nx2
pts2_px = m["pts2"].astype(np.float64)  # Nx2

img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)
assert img1 is not None and img2 is not None, "Immagini non lette: controlla i path."

# ======================
# Helpers
# ======================
def to_hom(pts2d):
    return np.hstack([pts2d, np.ones((pts2d.shape[0], 1), dtype=np.float64)])

def hartley_normalize_2d(x):
    """
    Hartley normalization (correct).
    Input: x Nx2 (inhomogeneous)
    Output: x_norm Nx2, T 3x3 such that x_norm ~ (T * [x;1]) / w
    """
    mu = np.mean(x, axis=0)
    x0 = x - mu
    d = np.sqrt(np.sum(x0**2, axis=1))
    mean_d = np.mean(d) + 1e-12
    s = np.sqrt(2.0) / mean_d

    T = np.array([
        [s, 0, -s*mu[0]],
        [0, s, -s*mu[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    x_h = to_hom(x)               # Nx3
    x_nh = (T @ x_h.T).T          # Nx3
    x_n = x_nh[:, :2] / x_nh[:, 2:3]
    return x_n, T

def eight_point_E(x1, x2, do_preconditioning=True):
    """
    x1,x2: Nx2 in normalized camera coords (undistorted + K^-1 already)
    Returns: E (3x3) with essential constraint enforced.
    """
    if do_preconditioning:
        x1n, T1 = hartley_normalize_2d(x1)
        x2n, T2 = hartley_normalize_2d(x2)
    else:
        x1n, x2n = x1, x2
        T1 = T2 = np.eye(3)

    x1x, x1y = x1n[:, 0], x1n[:, 1]
    x2x, x2y = x2n[:, 0], x2n[:, 1]

    # La tua matrice A originale: perfetta per x2^T * E * x1 = 0
    A = np.column_stack([
        x2x*x1x, x2x*x1y, x2x,
        x2y*x1x, x2y*x1y, x2y,
        x1x,     x1y,     np.ones_like(x1x)
    ])

    _, _, Vt = np.linalg.svd(A)
    E_tilde = Vt[-1].reshape(3, 3)

    # unnormalize if needed
    E = T2.T @ E_tilde @ T1

    # enforce essential constraint: singular values (s,s,0)
    U, S, Vt = np.linalg.svd(E)
    s = 0.5*(S[0] + S[1])
    E = U @ np.diag([s, s, 0.0]) @ Vt
    return E

def sampson_error(E, x1, x2):
    """
    Sampson distance for E on normalized camera coords.
    x1,x2: Nx2
    """
    x1h = to_hom(x1)
    x2h = to_hom(x2)

    Ex1 = (E @ x1h.T).T
    Etx2 = (E.T @ x2h.T).T
    x2tEx1 = np.sum(x2h * Ex1, axis=1)

    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2
    return (x2tEx1**2) / (denom + 1e-12)

def ransac_eight_point(x1, x2, iters=5000, threshold=1e-6, seed=0):
    """
    RANSAC around 8-point.
    threshold is in normalized coords (Sampson distance).
    Returns: best_E, best_mask (bool array)
    """
    rng = np.random.default_rng(seed)
    N = len(x1)
    assert N >= 8, "Servono almeno 8 match."

    best_E = None
    best_mask = None
    best_inliers = -1

    for _ in range(iters):
        idx = rng.choice(N, size=8, replace=False)
        E_cand = eight_point_E(x1[idx], x2[idx], do_preconditioning=True)

        err = sampson_error(E_cand, x1, x2)
        mask = err < threshold
        n_in = int(np.sum(mask))

        if n_in > best_inliers:
            best_inliers = n_in
            best_E = E_cand
            best_mask = mask

    # refinement with all inliers
    if best_mask is not None and np.sum(best_mask) >= 8:
        best_E = eight_point_E(x1[best_mask], x2[best_mask], do_preconditioning=True)

    return best_E, best_mask

def draw_epilines_on_img1(img1_u, pts2_u_px, F, out_path, max_lines=120):
    """
    Draw epipolar lines in image1 for points in image2:
    l1 = F^T x2
    pts2_u_px must be pixel coords in UNDISTORTED image system.
    """
    H, W = img1_u.shape[:2]
    vis = img1_u.copy()

    idx = np.arange(len(pts2_u_px))
    if max_lines is not None and len(idx) > max_lines:
        np.random.seed(0)
        idx = np.random.choice(idx, size=max_lines, replace=False)

    for i in idx:
        x2 = np.array([pts2_u_px[i,0], pts2_u_px[i,1], 1.0], dtype=np.float64)
        a, b, c = (F.T @ x2)

        pts = []
        if abs(b) > 1e-9:
            y0 = int(round((-c - a*0) / b))
            yW = int(round((-c - a*(W-1)) / b))
            pts = [(0, y0), (W-1, yW)]
        elif abs(a) > 1e-9:
            x = int(round(-c / a))
            pts = [(x, 0), (x, H-1)]

        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], (0,255,0), 1)

    cv2.imwrite(out_path, vis)

def point_line_dist_in_img1(F, pts1_u_px, pts2_u_px, mask=None):
    """
    Mean/median point-to-epiline distance in img1 (pixel, undistorted images).
    l1 = F^T x2; distance of x1 from l1.
    """
    if mask is None:
        mask = np.ones(len(pts1_u_px), dtype=bool)
    p1 = pts1_u_px[mask]
    p2 = pts2_u_px[mask]

    p1h = np.hstack([p1, np.ones((len(p1),1))])
    p2h = np.hstack([p2, np.ones((len(p2),1))])

    l1 = (F.T @ p2h.T).T  # Nx3
    num = np.abs(np.sum(l1 * p1h, axis=1))
    den = np.sqrt(l1[:,0]**2 + l1[:,1]**2) + 1e-12
    d = num/den
    return float(np.mean(d)), float(np.median(d))

# ======================
# 1) Undistort points -> normalized camera coords
# ======================
pts1_norm = cv2.undistortPoints(pts1_px.reshape(-1,1,2), K, dist).reshape(-1,2)
pts2_norm = cv2.undistortPoints(pts2_px.reshape(-1,1,2), K, dist).reshape(-1,2)

# ======================
# 2) Your 8-point + RANSAC
# ======================
# FIX: threshold=1e-6 (perché calcoliamo l'errore al quadrato nello spazio normalizzato!)
E_8r, mask_8r = ransac_eight_point(pts1_norm, pts2_norm, iters=5000, threshold=1e-6, seed=0)
assert E_8r is not None and mask_8r is not None, "RANSAC 8-point fallito."

err_8r = sampson_error(E_8r, pts1_norm, pts2_norm)

print("\n=== YOUR 8-POINT + RANSAC ===")
print("Inliers:", int(np.sum(mask_8r)), "/", len(mask_8r))
print("E_8_ransac =\n", E_8r)
print("Sampson mean (all):", float(np.mean(err_8r)))
print("Sampson median (all):", float(np.median(err_8r)))

# Convert to F for pixel drawing: F = K^-T E K^-1
F_8r = np.linalg.inv(K).T @ E_8r @ np.linalg.inv(K)

# ======================
# 3) Undistort images + pixel points in undistorted image system
# ======================
img1_u = cv2.undistort(img1, K, dist)
img2_u = cv2.undistort(img2, K, dist)
cv2.imwrite(os.path.join(OUT_DIR, "img1_undistorted.png"), img1_u)
cv2.imwrite(os.path.join(OUT_DIR, "img2_undistorted.png"), img2_u)

pts1_u_px = cv2.undistortPoints(pts1_px.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)
pts2_u_px = cv2.undistortPoints(pts2_px.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

draw_epilines_on_img1(
    img1_u,
    pts2_u_px[mask_8r],
    F_8r,
    out_path=os.path.join(OUT_DIR, "epilines_img1_8point_RANSAC.png"),
    max_lines=120
)

d8_mean, d8_med = point_line_dist_in_img1(F_8r, pts1_u_px, pts2_u_px, mask_8r)
print(f"8-point RANSAC epipolar dist in img1 (px) mean={d8_mean:.3f}, median={d8_med:.3f}")

# ======================
# 4) OpenCV 5-point (RANSAC) on normalized coords
# ======================
E_cv, mask_cv = cv2.findEssentialMat(
    pts1_norm.reshape(-1,1,2),
    pts2_norm.reshape(-1,1,2),
    np.eye(3),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1e-3
)
assert E_cv is not None and mask_cv is not None, "findEssentialMat fallito."
mask_cv = mask_cv.ravel().astype(bool)
if E_cv.shape[0] > 3:
    E_cv = E_cv[:3, :3]

err_cv = sampson_error(E_cv, pts1_norm[mask_cv], pts2_norm[mask_cv])

print("\n=== OPENCV 5-POINT (RANSAC) ===")
print("Inliers:", int(np.sum(mask_cv)), "/", len(mask_cv))
print("E_cv =\n", E_cv)
print("Sampson mean (on its inliers):", float(np.mean(err_cv)))
print("Sampson median (on its inliers):", float(np.median(err_cv)))

F_cv = np.linalg.inv(K).T @ E_cv @ np.linalg.inv(K)

draw_epilines_on_img1(
    img1_u,
    pts2_u_px[mask_cv],
    F_cv,
    out_path=os.path.join(OUT_DIR, "epilines_img1_opencv5pt.png"),
    max_lines=120
)

dcv_mean, dcv_med = point_line_dist_in_img1(F_cv, pts1_u_px, pts2_u_px, mask_cv)
print(f"5-point epipolar dist in img1 (px) mean={dcv_mean:.3f}, median={dcv_med:.3f}")

# ======================
# Save everything
# ======================
np.savez(os.path.join(OUT_DIR, "E_results.npz"),
         E_8=E_8r, F_8=F_8r, sampson_8=err_8r,
         E_cv=E_cv, F_cv=F_cv, sampson_cv=err_cv,
         mask_cv=mask_cv,
         pts1_px=pts1_px, pts2_px=pts2_px,
         pts1_norm=pts1_norm, pts2_norm=pts2_norm)

print("\nSalvati in outputs/:")
print(" - img1_undistorted.png, img2_undistorted.png")
print(" - epilines_img1_8point_RANSAC.png")
print(" - epilines_img1_opencv5pt.png")
print(" - E_results.npz")