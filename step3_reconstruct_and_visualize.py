import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG1_PATH = "object/img1.JPG"
CALIB_PATH = "camera_calib.npz"
E_RES_PATH = "outputs/E_results.npz"

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

img1 = cv2.imread(IMG1_PATH)
assert img1 is not None, "Errore caricamento img1."
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

cal = np.load(CALIB_PATH)
K = cal["K"].astype(np.float64)
dist = cal["dist"].astype(np.float64)

res = np.load(E_RES_PATH, allow_pickle=True)

# ============================================================
# MODIFICA 1: USA SOLO LA TUA SOLUZIONE (E_8 + mask_8)
# ============================================================
if "E_8" not in res:
    raise RuntimeError("E_8 non presente in E_results.npz: rigenera lo step2 salvando E_8.")

E = res["E_8"].astype(np.float64)

if "mask_8" not in res:
    raise RuntimeError(
        "mask_8 non presente in E_results.npz: non posso usare SOLO la tua E_8 in modo coerente.\n"
        "Rigenera E_results.npz con lo step2 che salva mask_8 (o modifica step2 per salvarla)."
    )

mask_in = res["mask_8"].astype(bool)
print("✅ Uso SOLO E_8 + mask_8 (nessun fallback su OpenCV)")

# ------------------------------------------------------------
# Caricamento punti
# ------------------------------------------------------------
if "pts1_px" not in res or "pts1_norm" not in res or "pts2_norm" not in res:
    raise RuntimeError(
        "Nel file E_results.npz mancano pts1_px / pts1_norm / pts2_norm.\n"
        "Assicurati che step2 salvi questi array."
    )

pts1_px = res["pts1_px"].astype(np.float64)
pts1_norm = res["pts1_norm"].astype(np.float64)
pts2_norm = res["pts2_norm"].astype(np.float64)

# opzionale: per distinguere punti manuali nel plot
src_is_manual = res["src_is_manual"].astype(bool) if "src_is_manual" in res else np.zeros(len(pts1_px), dtype=bool)

# Applica solo la tua mask_8
pts1_n = pts1_norm[mask_in].reshape(-1, 1, 2)
pts2_n = pts2_norm[mask_in].reshape(-1, 1, 2)
pts1_p = pts1_px[mask_in]
is_manual = src_is_manual[mask_in]

# ------------------------------------------------------------
# recoverPose con la tua E_8 (cameraMatrix = I perché punti norm)
# ------------------------------------------------------------
retval, R, t, mask_pose = cv2.recoverPose(E, pts1_n, pts2_n, np.eye(3))
mask_pose = mask_pose.ravel().astype(bool)

pts1_final_n = pts1_n[mask_pose].reshape(-1, 2)
pts2_final_n = pts2_n[mask_pose].reshape(-1, 2)
pts1_final_px = pts1_p[mask_pose].astype(np.int32)
is_manual_final = is_manual[mask_pose]

# ------------------------------------------------------------
# Triangolazione (coordinate normalizzate)
# ------------------------------------------------------------
P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = np.hstack([R, t])

X_h = cv2.triangulatePoints(P1, P2, pts1_final_n.T, pts2_final_n.T)
X = (X_h[:3, :] / (X_h[3, :] + 1e-12)).T

# Colori (nota: se pts1_px sono undistorti e img1 è distorta, i colori possono essere leggermente “shiftati”)
colors_bgr = np.array([img1_rgb[
    np.clip(pt[1], 0, img1_rgb.shape[0] - 1),
    np.clip(pt[0], 0, img1_rgb.shape[1] - 1)
] / 255.0 for pt in pts1_final_px])

# ------------------ FILTRI ------------------
X2 = (R @ X.T + t).T
mask_cheir = (X[:, 2] > 0) & (X2[:, 2] > 0)

dists_all = np.linalg.norm(X, axis=1)
th = np.quantile(dists_all[mask_cheir], 0.95) if np.sum(mask_cheir) > 0 else np.inf
mask_dist = dists_all < th

mask_keep = mask_cheir & mask_dist
X = X[mask_keep]
colors_bgr = colors_bgr[mask_keep]
is_manual_final = is_manual_final[mask_keep]

n_sift = int(np.sum(~is_manual_final))
n_manual = int(np.sum(is_manual_final))
print(f"Punti ricostruiti: {len(X)} (SIFT={n_sift}, MANUAL={n_manual})")

# ------------------ PLOT 3D ------------------
def transform_to_plot(pts):
    x_p = pts[:, 0]
    y_p = pts[:, 2]
    z_p = -pts[:, 1]
    return x_p, y_p, z_p

sift_mask = ~is_manual_final
manual_mask = is_manual_final

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

if np.sum(sift_mask) > 0:
    xs, ys, zs = transform_to_plot(X[sift_mask])
    ax.scatter(xs, ys, zs, c=colors_bgr[sift_mask], s=1, alpha=0.5, label=f"SIFT ({n_sift})")

if np.sum(manual_mask) > 0:
    xm, ym, zm = transform_to_plot(X[manual_mask])
    ax.scatter(xm, ym, zm, c="red", s=30, edgecolors="black", label=f"Manuali ({n_manual})")

ax.view_init(elev=5, azim=-90)
ax.set_xlabel("X (Sinistra-Destra)")
ax.set_ylabel("Profondità (Z camera)")
ax.set_zlabel("Altezza (Y camera)")

def safe_ptp(a, eps=1e-9):
    r = float(np.ptp(a))
    return r if r > eps else 1.0

x_all, y_all, z_all = transform_to_plot(X)
rx = safe_ptp(x_all)
ry = safe_ptp(y_all)

xmid = 0.5 * (x_all.min() + x_all.max())
ymid = 0.5 * (y_all.min() + y_all.max())
zmid = 0.5 * (z_all.min() + z_all.max())

ax.set_box_aspect([rx, ry, rx])
ax.set_xlim(xmid - rx / 2, xmid + rx / 2)
ax.set_ylim(ymid - ry / 2, ymid + ry / 2)
ax.set_zlim(zmid - rx / 2, zmid + rx / 2)

plt.title("Vista 3D allineata alla Fotocamera 1 (SOLO E_8 + mask_8)")
plt.legend()
plt.show()