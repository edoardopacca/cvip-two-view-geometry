import glob, os
import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False) 
# ===== PARAMETRI =====
pattern_size = (10, 7)      
square_size_mm = 24.0

img_paths = sorted(glob.glob("checkerboard/*.*"))
assert len(img_paths) > 0, "Nessuna immagine trovata in data/calib/"


cols, rows = pattern_size
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objp *= square_size_mm

objpoints, imgpoints = [], []
img_size = None

os.makedirs("debug_corners", exist_ok=True)

ok = 0
for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]  # (w, h)

    # detector robusto
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size)

    if not found:
        print(f"[NO] {os.path.basename(p)}")
        continue

    objpoints.append(objp)
    imgpoints.append(corners)
    ok += 1
    print(f"[OK] {os.path.basename(p)}  corners={len(corners)}")

    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    cv2.imwrite(os.path.join("debug_corners", os.path.basename(p)), vis)

print(f"\nTrovate scacchiere in {ok}/{len(img_paths)} immagini")
assert ok >= 8, "Troppo poche immagini valide: punta a 10–20."

# ===== CALIBRAZIONE =====
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

# errore medio di riproiezione
mean_err = 0.0
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_err += err
mean_err /= len(objpoints)

print("\n=== RISULTATI ===")
print("RMS (calibrateCamera):", rms)
print("Mean reprojection error (px):", mean_err)
print("\nK =\n", K)
print("\ndist =\n", dist.ravel())

np.savez("camera_calib.npz",
         K=K, dist=dist, rms=rms, mean_reproj_error=mean_err,
         img_size=np.array(img_size), pattern_size=np.array(pattern_size),
         square_size_mm=square_size_mm)
print("\nSalvato: camera_calib.npz")
print("Debug corners salvati in: debug_corners/")