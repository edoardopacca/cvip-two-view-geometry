import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG1_PATH = "object/img1.JPG"
CALIB_PATH = "camera_calib.npz"
E_RES_PATH = "outputs/E_results.npz"

OUT_DIR = "outputs"
MANUAL_GLOB = os.path.join(OUT_DIR, "manual_high_precision*.npz")
MANUAL_POINTS_PATH = os.path.join(OUT_DIR, "manual_high_precision.npz")

os.makedirs(OUT_DIR, exist_ok=True)

img1 = cv2.imread(IMG1_PATH)
assert img1 is not None, "Errore caricamento img1."
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

cal = np.load(CALIB_PATH)
K = cal["K"].astype(np.float64)
dist = cal["dist"].astype(np.float64)

res = np.load(E_RES_PATH)

# =======================
# USA LA TUA ESSENTIAL (8-point)
# =======================
E = res["E_8"].astype(np.float64)

# Se esiste una mask per il tuo RANSAC, usala. Altrimenti fallback su mask_cv.
if "mask_8" in res:
    mask_in = res["mask_8"].astype(bool)
else:
    # nel tuo file avevi salvato mask_cv ma non mask_8: fallback
    mask_in = res["mask_cv"].astype(bool)
    print("⚠️ Nota: 'mask_8' non trovata in E_results.npz, uso mask_cv come filtro punti.")

pts1_px = res["pts1_px"].astype(np.float64)
pts1_norm = res["pts1_norm"].astype(np.float64)
pts2_norm = res["pts2_norm"].astype(np.float64)

# punti normalizzati (undistorti) per recoverPose/triangolazione
pts1_n = pts1_norm[mask_in].reshape(-1, 1, 2)
pts2_n = pts2_norm[mask_in].reshape(-1, 1, 2)
pts1_p = pts1_px[mask_in]

retval, R, t, mask_pose = cv2.recoverPose(E, pts1_n, pts2_n, np.eye(3))
mask_pose = mask_pose.ravel().astype(bool)

pts1_final_n = pts1_n[mask_pose].reshape(-1, 2)
pts2_final_n = pts2_n[mask_pose].reshape(-1, 2)
pts1_final_px = pts1_p[mask_pose].astype(np.int32)

P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = np.hstack([R, t])

X_h = cv2.triangulatePoints(P1, P2, pts1_final_n.T, pts2_final_n.T)
X = (X_h[:3, :] / (X_h[3, :] + 1e-12)).T

# colori presi da img1 originale (pixel distorti)
# (se vuoi colori perfetti su undistorto, si può adattare)
colors = np.array([img1_rgb[pt[1], pt[0]] / 255.0 for pt in pts1_final_px])

# ------------------ PUNTI MANUALI: CARICA TUTTI I FILE ------------------
def load_all_manual_files(out_dir):
    files = sorted(glob.glob(MANUAL_GLOB))
    if os.path.exists(MANUAL_POINTS_PATH) and MANUAL_POINTS_PATH not in files:
        files.append(MANUAL_POINTS_PATH)
    files = [f for f in files if os.path.basename(f).startswith("manual_high_precision")]
    return sorted(set(files))

manual_files = load_all_manual_files(OUT_DIR)

X_manual = None
if len(manual_files) > 0:
    X_list = []
    total_pts = 0

    for f in manual_files:
        try:
            m_data = np.load(f)
            if "pts1" not in m_data or "pts2" not in m_data:
                print(f"⚠️ Skip {os.path.basename(f)}: manca pts1/pts2.")
                continue

            m_p1 = m_data["pts1"]
            m_p2 = m_data["pts2"]

            n = min(len(m_p1), len(m_p2))
            if n == 0:
                print(f"⚠️ Skip {os.path.basename(f)}: 0 punti.")
                continue
            if len(m_p1) != len(m_p2):
                print(f"⚠️ {os.path.basename(f)}: punti non accoppiati ({len(m_p1)} vs {len(m_p2)}). Uso i primi {n}.")

            m_p1 = m_p1[:n].astype(np.float64)
            m_p2 = m_p2[:n].astype(np.float64)

            # Se i punti manuali sono stati raccolti con il tool UNDISTORT:
            # - sono già pixel undistorti -> per normalizzarli basta K e dist=None/zeros
            und_flag = int(m_data["undistorted"][0]) if "undistorted" in m_data else 0

            if und_flag == 1:
                m_p1_n = cv2.undistortPoints(m_p1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
                m_p2_n = cv2.undistortPoints(m_p2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
            else:
                # tool vecchio (distorto): serve anche dist
                m_p1_n = cv2.undistortPoints(m_p1.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
                m_p2_n = cv2.undistortPoints(m_p2.reshape(-1, 1, 2), K, dist).reshape(-1, 2)

            m_X_h = cv2.triangulatePoints(P1, P2, m_p1_n.T, m_p2_n.T)
            m_X = (m_X_h[:3, :] / (m_X_h[3, :] + 1e-12)).T

            X_list.append(m_X)
            total_pts += len(m_X)
            print(f"✅ Caricati {len(m_X)} punti da {os.path.basename(f)}")
        except Exception as e:
            print(f"⚠️ Skip {os.path.basename(f)}: errore lettura ({e})")

    if len(X_list) > 0:
        X_manual = np.vstack(X_list)
        print(f"🎯 Totale punti manuali triangolati: {total_pts} (da {len(X_list)} file)")
    else:
        print("⚠️ Nessun punto manuale valido trovato.")
else:
    print("ℹ️ Nessun file manual_high_precision*.npz trovato in outputs/")

# ------------------ FILTRI SUI PUNTI SIFT ------------------
# cheirality: davanti alla camera 1 e camera 2
X2 = (R @ X.T + t).T
mask_cheir = (X[:, 2] > 0) & (X2[:, 2] > 0)
X = X[mask_cheir]
colors = colors[mask_cheir]

dists = np.linalg.norm(X, axis=1)
th = np.quantile(dists, 0.95)
mask_dist = dists < th
X = X[mask_dist]
colors = colors[mask_dist]

def transform_to_plot(pts):
    x_p = pts[:, 0]
    y_p = pts[:, 2]
    z_p = -pts[:, 1]
    return x_p, y_p, z_p

x_s, y_s, z_s = transform_to_plot(X)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x_s, y_s, z_s, c=colors, s=1, alpha=0.5, label="Punti SIFT")

if X_manual is not None:
    xm, ym, zm = transform_to_plot(X_manual)
    ax.scatter(xm, ym, zm, c="red", s=30, edgecolors="black", label="Punti Manuali")

ax.view_init(elev=5, azim=-90)

ax.set_xlabel("X (Sinistra-Destra)")
ax.set_ylabel("Profondità (Z camera)")
ax.set_zlabel("Altezza (Y camera)")

def safe_ptp(a, eps=1e-9):
    r = float(np.ptp(a))
    return r if r > eps else 1.0

x_all, y_all, z_all = x_s, y_s, z_s
if X_manual is not None:
    x_all = np.concatenate([x_s, xm])
    y_all = np.concatenate([y_s, ym])
    z_all = np.concatenate([z_s, zm])

rx = safe_ptp(x_all)
ry = safe_ptp(y_all)

xmid = 0.5 * (x_all.min() + x_all.max())
ymid = 0.5 * (y_all.min() + y_all.max())
zmid = 0.5 * (z_all.min() + z_all.max())

ax.set_box_aspect([rx, ry, rx])
ax.set_xlim(xmid - rx / 2, xmid + rx / 2)
ax.set_ylim(ymid - ry / 2, ymid + ry / 2)
ax.set_zlim(zmid - rx / 2, zmid + rx / 2)

plt.title("Vista 3D allineata alla Fotocamera 1 (E_8)")
plt.legend()
plt.show()