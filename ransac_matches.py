import os
import cv2
import numpy as np

img1_path = "object/img1.JPG"
img2_path = "object/img2.JPG"

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
assert img1 is not None and img2 is not None, "Errore: immagini non lette. Controlla i path."

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

use_sift = hasattr(cv2, "SIFT_create")
if use_sift:
    feat = cv2.SIFT_create(nfeatures=10000)
    norm = cv2.NORM_L2
else:
    print("SIFT non disponibile: uso ORB (meno robusto).")
    feat = cv2.ORB_create(nfeatures=4000)
    norm = cv2.NORM_HAMMING

kp1, des1 = feat.detectAndCompute(gray1, None)
kp2, des2 = feat.detectAndCompute(gray2, None)
assert des1 is not None and des2 is not None, "Nessun descrittore trovato."

bf = cv2.BFMatcher(norm, crossCheck=False)
matches_knn = bf.knnMatch(des1, des2, k=2)

ratio = 0.75 if use_sift else 0.80
good = []
for m, n in matches_knn:
    if m.distance < ratio * n.distance:
        good.append(m)

print(f"Keypoints: img1={len(kp1)} img2={len(kp2)}")
print(f"Matches dopo ratio test: {len(good)}")
assert len(good) >= 12, "Troppi pochi match: prova a cambiare ratio o usa immagini con più texture."

# Punti corrispondenti in pixel (DISTORTI, dalle immagini originali)
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

# Visualizzazione di tutti i match post-ratio-test (senza RANSAC)
vis = cv2.drawMatches(
    img1, kp1, img2, kp2,
    good, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(os.path.join(out_dir, "good_matches.png"), vis)

# Salva tutti i match post-ratio-test (senza filtraggio RANSAC)
np.savez(
    os.path.join(out_dir, "matches_px.npz"),
    pts1=pts1,
    pts2=pts2,
    undistorted=np.array([0], dtype=np.int32),   # esplicito: pixel distorti
    use_sift=np.array([1 if use_sift else 0], dtype=np.int32),
    ratio=np.array([ratio], dtype=np.float64),
)

print("Salvati:")
print(" - outputs/good_matches.png")
print(" - outputs/matches_px.npz")