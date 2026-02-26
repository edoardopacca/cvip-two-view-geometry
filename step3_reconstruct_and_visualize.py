import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG1_PATH = "object/img1.png"
CALIB_PATH = "camera_calib.npz"
E_RES_PATH = "outputs/E_results.npz"
MANUAL_POINTS_PATH = "outputs/manual_high_precision.npz" 
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

img1 = cv2.imread(IMG1_PATH)
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

cal = np.load(CALIB_PATH)
K = cal["K"].astype(np.float64)
dist = cal["dist"].astype(np.float64)

res = np.load(E_RES_PATH)
E = res["E_cv"].astype(np.float64)
mask_cv = res["mask_cv"].astype(bool)

pts1_px = res["pts1_px"].astype(np.float64)
pts1_norm = res["pts1_norm"].astype(np.float64)
pts2_norm = res["pts2_norm"].astype(np.float64)

pts1_n = pts1_norm[mask_cv].reshape(-1, 1, 2)
pts2_n = pts2_norm[mask_cv].reshape(-1, 1, 2)
pts1_p = pts1_px[mask_cv] 

retval, R, t, mask_pose = cv2.recoverPose(E, pts1_n, pts2_n, np.eye(3))
mask_pose = mask_pose.ravel().astype(bool)

pts1_final_n = pts1_n[mask_pose].reshape(-1, 2)
pts2_final_n = pts2_n[mask_pose].reshape(-1, 2)
pts1_final_px = pts1_p[mask_pose].astype(np.int32)

P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = np.hstack([R, t])

X_h = cv2.triangulatePoints(P1, P2, pts1_final_n.T, pts2_final_n.T)
X = (X_h[:3, :] / (X_h[3, :] + 1e-12)).T 

colors = np.array([img1_rgb[pt[1], pt[0]] / 255.0 for pt in pts1_final_px])

X_manual = None
if os.path.exists(MANUAL_POINTS_PATH):
    m_data = np.load(MANUAL_POINTS_PATH)
    m_p1, m_p2 = m_data['pts1'], m_data['pts2']
    
    m_p1_n = cv2.undistortPoints(m_p1.reshape(-1,1,2), K, dist).reshape(-1,2)
    m_p2_n = cv2.undistortPoints(m_p2.reshape(-1,1,2), K, dist).reshape(-1,2)
    
    m_X_h = cv2.triangulatePoints(P1, P2, m_p1_n.T, m_p2_n.T)
    X_manual = (m_X_h[:3, :] / (m_X_h[3, :] + 1e-12)).T
    print(f"✅ Inseriti {len(X_manual)} punti manuali della scritta.")

mask_z = X[:, 2] > 0
X = X[mask_z]
colors = colors[mask_z]

dists = np.linalg.norm(X, axis=1)
X = X[dists < np.quantile(dists, 0.95)]
colors = colors[dists < np.quantile(dists, 0.95)]

def transform_to_plot(pts):
    x_p = pts[:, 0]
    y_p = pts[:, 2]  
    z_p = -pts[:, 1] 
    return x_p, y_p, z_p

x_s, y_s, z_s = transform_to_plot(X)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_s, y_s, z_s, c=colors, s=1, alpha=0.5, label='Punti SIFT')

if X_manual is not None:
    xm, ym, zm = transform_to_plot(X_manual)
    ax.scatter(xm, ym, zm, c='red', s=30, edgecolors='black', label='Scritta (Manuale)')
    ax.plot(xm, ym, zm, c='red', linewidth=2, label='Outline Lettere')

ax.view_init(elev=5, azim=-90)

ax.set_xlabel('X (Sinistra-Destra)')
ax.set_ylabel('Profondità (Z camera)')
ax.set_zlabel('Altezza (Y camera)')

ax.set_box_aspect([np.ptp(x_s), np.ptp(y_s), np.ptp(z_s)]) 

plt.title("Vista 3D allineata alla Fotocamera 1")
plt.legend()
plt.show()