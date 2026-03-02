import os
import glob
import numpy as np
import cv2

CALIB_PATH   = "camera_calib.npz"
MATCHES_PATH = "outputs/matches_inliers_px.npz"
IMG1_PATH    = "object/img1.JPG"
IMG2_PATH    = "object/img2.JPG"

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANSAC_ITERS = 5000
RANSAC_THRESH = 1e-6
RANSAC_SEED = 0

cal = np.load(CALIB_PATH)
K = cal["K"].astype(np.float64)
dist = cal["dist"].astype(np.float64)

m = np.load(MATCHES_PATH, allow_pickle=True)

pts1_in = m["pts1"].astype(np.float64)
pts2_in = m["pts2"].astype(np.float64)

matches_undistorted = int(m["undistorted"][0]) if "undistorted" in m else 0

if matches_undistorted == 1:
    pts1_sift_u_px = pts1_in
    pts2_sift_u_px = pts2_in
else:
    pts1_sift_u_px = cv2.undistortPoints(pts1_in.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)
    pts2_sift_u_px = cv2.undistortPoints(pts2_in.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)
assert img1 is not None and img2 is not None, "Immagini non lette: controlla i path."

img1_u = cv2.undistort(img1, K, dist)
img2_u = cv2.undistort(img2, K, dist)
cv2.imwrite(os.path.join(OUT_DIR, "img1_undistorted.png"), img1_u)
cv2.imwrite(os.path.join(OUT_DIR, "img2_undistorted.png"), img2_u)

def load_all_manual_points(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "manual_high_precision*.npz")))
    pts1_list, pts2_list = [], []
    used_files = []

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            if "pts1" not in d or "pts2" not in d:
                continue
            p1 = d["pts1"].astype(np.float64)
            p2 = d["pts2"].astype(np.float64)

            n = min(len(p1), len(p2))
            if n < 1:
                continue
            p1 = p1[:n]
            p2 = p2[:n]

            und_flag = int(d["undistorted"][0]) if "undistorted" in d else 0
            if und_flag == 1:
                p1_u = p1
                p2_u = p2
            else:
                p1_u = cv2.undistortPoints(p1.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)
                p2_u = cv2.undistortPoints(p2.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

            pts1_list.append(p1_u)
            pts2_list.append(p2_u)
            used_files.append(os.path.basename(f))
        except Exception as e:
            print(f"⚠️ Skip {os.path.basename(f)}: {e}")

    if len(pts1_list) == 0:
        return None, None, used_files

    return np.vstack(pts1_list), np.vstack(pts2_list), used_files

MANUAL_DIR = "manual_points"
pts1_man_u_px, pts2_man_u_px, manual_files_used = load_all_manual_points(MANUAL_DIR)

if pts1_man_u_px is not None:
    pts1_u_px = np.vstack([pts1_sift_u_px, pts1_man_u_px])
    pts2_u_px = np.vstack([pts2_sift_u_px, pts2_man_u_px])
    src_is_manual = np.hstack([
        np.zeros(len(pts1_sift_u_px), dtype=bool),
        np.ones(len(pts1_man_u_px), dtype=bool),
    ])
    print(f"Manual points aggiunti: {len(pts1_man_u_px)} (da {len(manual_files_used)} file)")
else:
    pts1_u_px = pts1_sift_u_px
    pts2_u_px = pts2_sift_u_px
    src_is_manual = np.zeros(len(pts1_u_px), dtype=bool)
    print("Nessun file manual_high_precision*.npz trovato: uso solo SIFT/ORB.")

print(f"Totale corrispondenze usate (SIFT+MANUAL): {len(pts1_u_px)}")
assert len(pts1_u_px) >= 8, "Servono almeno 8 corrispondenze totali (SIFT+manual)."

pts1_norm = cv2.undistortPoints(pts1_u_px.reshape(-1,1,2), K, None).reshape(-1,2)
pts2_norm = cv2.undistortPoints(pts2_u_px.reshape(-1,1,2), K, None).reshape(-1,2)

def to_hom(pts2d):
    return np.hstack([pts2d, np.ones((pts2d.shape[0], 1), dtype=np.float64)])

def hartley_normalize_2d(x):
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

    x_h = to_hom(x)
    x_nh = (T @ x_h.T).T
    x_n = x_nh[:, :2] / x_nh[:, 2:3]
    return x_n, T

def eight_point_E(x1, x2, do_preconditioning=True):
    if do_preconditioning:
        x1n, T1 = hartley_normalize_2d(x1)
        x2n, T2 = hartley_normalize_2d(x2)
    else:
        x1n, x2n = x1, x2
        T1 = T2 = np.eye(3)

    x1x, x1y = x1n[:, 0], x1n[:, 1]
    x2x, x2y = x2n[:, 0], x2n[:, 1]

    A = np.column_stack([
        x2x*x1x, x2x*x1y, x2x,
        x2y*x1x, x2y*x1y, x2y,
        x1x,     x1y,     np.ones_like(x1x)
    ])

    _, _, Vt = np.linalg.svd(A)
    E_tilde = Vt[-1].reshape(3, 3)

    E = T2.T @ E_tilde @ T1

    U, S, Vt = np.linalg.svd(E)
    s = 0.5*(S[0] + S[1])
    E = U @ np.diag([s, s, 0.0]) @ Vt
    return E

def sampson_error(E, x1, x2):
    x1h = to_hom(x1)
    x2h = to_hom(x2)

    Ex1 = (E @ x1h.T).T
    Etx2 = (E.T @ x2h.T).T
    x2tEx1 = np.sum(x2h * Ex1, axis=1)

    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2
    return (x2tEx1**2) / (denom + 1e-12)

def ransac_eight_point(x1, x2, iters=5000, threshold=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    N = len(x1)
    assert N >= 8, "Servono almeno 8 match."

    best_E = None
    best_mask = None
    best_inliers = -1
    best_score = np.inf

    for _ in range(iters):
        idx = rng.choice(N, size=8, replace=False)
        E_cand = eight_point_E(x1[idx], x2[idx], do_preconditioning=True)

        err = sampson_error(E_cand, x1, x2)
        mask = err < threshold
        n_in = int(np.sum(mask))
        if n_in < 8:
            continue

        score = float(np.mean(err[mask]))

        if (n_in > best_inliers) or (n_in == best_inliers and score < best_score):
            best_inliers = n_in
            best_E = E_cand
            best_mask = mask
            best_score = score

    assert best_E is not None and best_mask is not None, "RANSAC non ha trovato un modello valido."

    E_refit = eight_point_E(x1[best_mask], x2[best_mask], do_preconditioning=True)

    err_refit = sampson_error(E_refit, x1, x2)
    mask_refit = err_refit < threshold

    if int(np.sum(mask_refit)) >= 8 and int(np.sum(mask_refit)) >= int(best_inliers * 0.8):
        return E_refit, mask_refit
    else:
        return best_E, best_mask

def draw_epilines_on_img1(img1_u, pts2_u_px, F, out_path, max_lines=120):
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
    if mask is None:
        mask = np.ones(len(pts1_u_px), dtype=bool)
    p1 = pts1_u_px[mask]
    p2 = pts2_u_px[mask]

    p1h = np.hstack([p1, np.ones((len(p1),1))])
    p2h = np.hstack([p2, np.ones((len(p2),1))])

    l1 = (F.T @ p2h.T).T
    num = np.abs(np.sum(l1 * p1h, axis=1))
    den = np.sqrt(l1[:,0]**2 + l1[:,1]**2) + 1e-12
    d = num/den
    return float(np.mean(d)), float(np.median(d))

E_8r, mask_8r = ransac_eight_point(
    pts1_norm, pts2_norm,
    iters=RANSAC_ITERS,
    threshold=RANSAC_THRESH,
    seed=RANSAC_SEED
)
assert E_8r is not None and mask_8r is not None, "RANSAC 8-point fallito."

err_8_all = sampson_error(E_8r, pts1_norm, pts2_norm)

print("\n=== YOUR 8-POINT + RANSAC (SIFT + MANUAL) ===")
print("Inliers:", int(np.sum(mask_8r)), "/", len(mask_8r))
print("E_8_ransac =\n", E_8r)
print("Sampson mean (inliers):", float(np.mean(err_8_all[mask_8r])))
print("Sampson median (inliers):", float(np.median(err_8_all[mask_8r])))

n_in_manual = int(np.sum(mask_8r & src_is_manual))
n_in_sift = int(np.sum(mask_8r & (~src_is_manual)))
print(f"Inlier breakdown: SIFT={n_in_sift}, MANUAL={n_in_manual}")

F_8r = np.linalg.inv(K).T @ E_8r @ np.linalg.inv(K)

draw_epilines_on_img1(
    img1_u,
    pts2_u_px[mask_8r],
    F_8r,
    out_path=os.path.join(OUT_DIR, "epilines_img1_8point_RANSAC.png"),
    max_lines=120
)

d8_mean, d8_med = point_line_dist_in_img1(F_8r, pts1_u_px, pts2_u_px, mask_8r)
print(f"8-point epipolar dist in img1 (px) mean={d8_mean:.3f}, median={d8_med:.3f}")

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

err_cv_all = sampson_error(E_cv, pts1_norm, pts2_norm)

print("\n=== OPENCV 5-POINT (RANSAC) ===")
print("Inliers:", int(np.sum(mask_cv)), "/", len(mask_cv))
print("E_cv =\n", E_cv)
print("Sampson mean (inliers):", float(np.mean(err_cv_all[mask_cv])))
print("Sampson median (inliers):", float(np.median(err_cv_all[mask_cv])))

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

np.savez(
    os.path.join(OUT_DIR, "E_results.npz"),
    E_8=E_8r, F_8=F_8r, sampson_8=err_8_all,
    E_cv=E_cv, F_cv=F_cv, sampson_cv=err_cv_all,
    mask_8=mask_8r.astype(bool),
    mask_cv=mask_cv.astype(bool),
    pts1_px=pts1_u_px.astype(np.float64),
    pts2_px=pts2_u_px.astype(np.float64),
    pts1_norm=pts1_norm.astype(np.float64),
    pts2_norm=pts2_norm.astype(np.float64),
    src_is_manual=src_is_manual.astype(bool),
    manual_files=np.array(manual_files_used, dtype=object)
)

print("\nSalvati in outputs/:")
print(" - img1_undistorted.png, img2_undistorted.png")
print(" - epilines_img1_8point_RANSAC.png")
print(" - epilines_img1_opencv5pt.png")
print(" - E_results.npz  (include mask_8 + SIFT+MANUAL)")