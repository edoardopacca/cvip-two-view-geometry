import cv2
import numpy as np
import os
from datetime import datetime

# ============================================================
#  MANUAL POINTS TOOL (DISTORTED MODE) - SINGLE FILE
#  - Mostra img1/img2 ORIGINALI (DISTORTE)
#  - I punti salvati (pts1/pts2) sono PIXEL DISTORTI (coordinate immagine originale)
#  - Salva SOLO 1 file: manual_high_precision_<SESSION_ID>.npz (autosave continuo)
#  - Disegna linea epipolare sulla foto destra quando selezioni un punto sulla sinistra
#    (basata su F stimata da OpenCV su immagini DISTORTE)
#
#  ✅ NOTA per Step 2:
#  Tratta questi punti come i SIFT (distorti):
#     p_norm = cv2.undistortPoints(p_px.reshape(-1,1,2), K, dist).reshape(-1,2)
# ============================================================

# ------------------ PATHS ------------------
IMG1_PATH = "object/img1.JPG"
IMG2_PATH = "object/img2.JPG"
CALIB_PATH = "camera_calib.npz"

OUT_DIR = "outputs"
MANUAL_DIR = "manual_points"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MANUAL_DIR, exist_ok=True)

# Unica sessione = un unico file, sempre aggiornato
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
FINAL_FILE = os.path.join(MANUAL_DIR, f"manual_high_precision_{SESSION_ID}.npz")

print(f"🆔 Sessione: {SESSION_ID}")
print(f"💾 File unico (autosave): {FINAL_FILE}")

# ------------------ LOAD CALIB ------------------
cal = np.load(CALIB_PATH)
K = cal["K"].astype(np.float64)
dist = cal["dist"].astype(np.float64)

# ------------------ LOAD IMAGES (DISTORTED) ------------------
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)
assert img1 is not None and img2 is not None, "Errore caricamento immagini."

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
assert h1 == h2, "Le immagini devono avere la stessa altezza per hconcat."

combo_orig = cv2.hconcat([img1, img2])
combo_disp = combo_orig.copy()

# ============================================================
#  EPIPOLAR: stima F (DISTORTED) con OpenCV (SIFT/ORB + RANSAC)
# ============================================================
def estimate_F_opencv_distorted(img1_bgr, img2_bgr):
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    use_sift = hasattr(cv2, "SIFT_create")
    if use_sift:
        feat = cv2.SIFT_create(nfeatures=8000)
        norm = cv2.NORM_L2
        ratio = 0.75
    else:
        print("SIFT non disponibile: uso ORB (meno robusto per F).")
        feat = cv2.ORB_create(nfeatures=4000)
        norm = cv2.NORM_HAMMING
        ratio = 0.80

    kp1, des1 = feat.detectAndCompute(gray1, None)
    kp2, des2 = feat.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 20:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC su F nel dominio DISTORTO (coerente con ciò che visualizzi)
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999,
        maxIters=5000
    )
    if F is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)
    if int(np.sum(mask)) < 8:
        return None

    # Refinement: ricalcola F con 8-point sui soli inlier (più “pulito”)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]
    F_ref, _ = cv2.findFundamentalMat(pts1_in, pts2_in, method=cv2.FM_8POINT)
    if F_ref is not None:
        F = F_ref

    # normalizza scala per stabilità numerica
    F = F / (np.linalg.norm(F) + 1e-12)
    return F

print("🔎 Stimo F (OpenCV) per disegnare epipolari (su immagini DISTORTE)...")
F_epi = estimate_F_opencv_distorted(img1, img2)
if F_epi is None:
    print("⚠️ Non sono riuscito a stimare F: epipolar line OFF (puoi comunque cliccare e salvare punti).")
else:
    print("✅ F stimata: epipolar line ON")

def compute_epiline_right_from_left(pt_left):
    """pt_left: (u,v) su immagine sinistra DISTORTA. Ritorna linea su destra ax+by+c=0."""
    if F_epi is None:
        return None
    x1 = np.array([pt_left[0], pt_left[1], 1.0], dtype=np.float64)
    l2 = F_epi @ x1
    a, b, c = l2
    n = np.sqrt(a*a + b*b) + 1e-12
    return np.array([a/n, b/n, c/n], dtype=np.float64)

def line_segment_in_image(a, b, c, W, H):
    pts = []
    if abs(b) > 1e-12:
        y0 = -(c + a * 0) / b
        yW = -(c + a * (W - 1)) / b
        if 0 <= y0 <= H - 1: pts.append((0, int(round(y0))))
        if 0 <= yW <= H - 1: pts.append((W - 1, int(round(yW))))
    if abs(a) > 1e-12:
        x0 = -(c + b * 0) / a
        xH = -(c + b * (H - 1)) / a
        if 0 <= x0 <= W - 1: pts.append((int(round(x0)), 0))
        if 0 <= xH <= W - 1: pts.append((int(round(xH)), H - 1))

    if len(pts) >= 2:
        best = (pts[0], pts[1])
        best_d = -1
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d = dx * dx + dy * dy
                if d > best_d:
                    best_d = d
                    best = (pts[i], pts[j])
        return best[0], best[1]
    return None, None

def draw_epiline_on_right(vis, line_right, thickness=2):
    """Disegna la retta epipolare nella metà destra (offset w1)."""
    if line_right is None:
        return
    a, b, c = line_right
    p0, p1 = line_segment_in_image(a, b, c, w2, h2)
    if p0 is None:
        return
    p0d = (p0[0] + w1, p0[1])
    p1d = (p1[0] + w1, p1[1])
    cv2.line(vis, p0d, p1d, (0, 255, 255), thickness)

# ------------------ STATE ------------------
pts1 = []
pts2 = []
expecting = 1  # 1 = click sinistra, 2 = click destra

cursor_x, cursor_y = w1 // 2, h1 // 2
zoom_factor = 6
patch_radius = 30

current_epi_line_right = None  # linea epipolare sulla foto destra (quando scelgo punto sx)

# ------------------ SAVE HELPERS ------------------
def atomic_save_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)

def save_state():
    atomic_save_npz(
        FINAL_FILE,
        pts1=np.array(pts1, dtype=np.float32),
        pts2=np.array(pts2, dtype=np.float32),
        expecting=np.array([expecting], dtype=np.int32),
        cursor=np.array([cursor_x, cursor_y], dtype=np.int32),
        w1=np.array([w1], dtype=np.int32),
        img1=np.array([IMG1_PATH]),
        img2=np.array([IMG2_PATH]),
        session_id=np.array([SESSION_ID]),
        undistorted=np.array([0], dtype=np.int32),  # 0 = DISTORTI
        K=K,
        dist=dist,
        is_complete=np.array([1 if (len(pts1) == len(pts2) and len(pts1) > 0) else 0], dtype=np.int32),
    )

# ------------------ DRAWING ------------------
def redraw():
    global combo_disp
    combo_disp = combo_orig.copy()

    # punti su sinistra (rossi)
    for i, pt in enumerate(pts1):
        cv2.circle(combo_disp, pt, 4, (0, 0, 255), -1)
        cv2.putText(combo_disp, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # punti su destra (verdi) + linee match
    for i, pt in enumerate(pts2):
        pt_disp = (int(pt[0] + w1), int(pt[1]))
        cv2.circle(combo_disp, pt_disp, 4, (0, 255, 0), -1)
        cv2.putText(combo_disp, str(i + 1), (pt_disp[0] + 8, pt_disp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if i < len(pts1):
            cv2.line(combo_disp, pts1[i], pt_disp, (255, 255, 0), 1)

    # ✅ linea epipolare sulla destra quando stai scegliendo il punto dx
    if expecting == 2 and current_epi_line_right is not None:
        draw_epiline_on_right(combo_disp, current_epi_line_right, thickness=2)

    msg = "1. FOTO SINISTRA (DISTORTED)" if expecting == 1 else "2. FOTO DESTRA (DISTORTED)"
    color = (0, 0, 255) if expecting == 1 else (0, 255, 0)
    cv2.putText(combo_disp, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    info = f"SX:{len(pts1)}  DX:{len(pts2)}  | click: {'SX' if expecting==1 else 'DX'}"
    cv2.putText(combo_disp, info, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if F_epi is None:
        cv2.putText(combo_disp, "F non stimata: epipolar line OFF", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# --------- LENTE ----------
def update_lens(frame_for_lens):
    ch, cw = frame_for_lens.shape[:2]
    y_min, y_max = max(0, cursor_y - patch_radius), min(ch, cursor_y + patch_radius)
    x_min, x_max = max(0, cursor_x - patch_radius), min(cw, cursor_x + patch_radius)

    patch = frame_for_lens[y_min:y_max, x_min:x_max].copy()
    if patch.shape[0] > 0 and patch.shape[1] > 0:
        zoomed = cv2.resize(
            patch, (0, 0),
            fx=zoom_factor, fy=zoom_factor,
            interpolation=cv2.INTER_NEAREST
        )
        zh, zw = zoomed.shape[:2]
        cv2.line(zoomed, (zw // 2, 0), (zw // 2, zh), (255, 0, 0), 1)
        cv2.line(zoomed, (0, zh // 2), (zw, zh // 2), (255, 0, 0), 1)
        cv2.imshow("Lente", zoomed)

def update_current_epiline():
    global current_epi_line_right
    if expecting == 2 and len(pts1) > 0:
        current_epi_line_right = compute_epiline_right_from_left(pts1[-1])
    else:
        current_epi_line_right = None

def save_point():
    global expecting
    x, y = cursor_x, cursor_y

    if expecting == 1 and x < w1:
        pts1.append((x, y))
        expecting = 2
        update_current_epiline()
        print(f"✅ Punto {len(pts1)} registrato su SINISTRA (DISTORTED). (epiline -> DESTRA)")
    elif expecting == 2 and x >= w1:
        pts2.append((x - w1, y))
        expecting = 1
        update_current_epiline()
        print(f"✅ Punto {len(pts2)} registrato su DESTRA (DISTORTED).")
    else:
        print("⚠️ Cursore fuori posto! Spostalo nell'immagine corretta.")

    redraw()
    save_state()

def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        save_point()

# ------------------ UI SETUP ------------------
cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main", 1600, 600)
cv2.namedWindow("Lente", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Main", mouse_callback)

redraw()
save_state()

print("\nControlli:")
print("  - Click sinistro: salva punto (alternando SX -> DX)")
print("  - Spazio: salva punto (come click)")
print("  - WASD / frecce: muovi cursore di 1px")
print("  - z: undo ultimo punto (gestisce SX/DX)")
print("  - f: stampa stato (il file è già autosalvato)")
print("  - q o ESC: esci (autosave sempre)\n")

# ------------------ MAIN LOOP ------------------
try:
    while True:
        temp_disp = combo_disp.copy()
        update_lens(temp_disp)
        cv2.imshow("Main", temp_disp)

        key = cv2.waitKeyEx(15)
        if key == -1:
            continue

        if key == ord('q') or key == 27:
            break

        elif key == ord('z'):
            if expecting == 2:
                if len(pts1) > 0:
                    pts1.pop()
                    expecting = 1
                    update_current_epiline()
                    print("↩️ Undo: Punto SINISTRA rimosso.")
            elif expecting == 1 and len(pts2) > 0:
                pts2.pop()
                expecting = 2
                update_current_epiline()
                print("↩️ Undo: Punto DESTRA rimosso.")
            redraw()
            save_state()

        elif key == ord(' '):
            save_point()

        elif key == ord('f'):
            save_state()
            if len(pts1) == len(pts2) and len(pts1) > 0:
                print(f"🎉 OK: punti pari. File unico aggiornato: {FINAL_FILE}  ({len(pts1)} coppie)")
            else:
                print("⚠️ Punti non pari o zero: il file è salvato comunque (Step2 userà solo le coppie complete).")

        elif key == ord('w') or key in [63232, 2490368, 82]:
            cursor_y -= 1
        elif key == ord('s') or key in [63233, 2621440, 84]:
            cursor_y += 1
        elif key == ord('a') or key in [63234, 2424832, 81]:
            cursor_x -= 1
        elif key == ord('d') or key in [63235, 2555904, 83]:
            cursor_x += 1

        cursor_x = max(0, min(cursor_x, combo_orig.shape[1] - 1))
        cursor_y = max(0, min(cursor_y, combo_orig.shape[0] - 1))

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()

save_state()
print(f"\n💾 File unico salvato/aggiornato in: {FINAL_FILE}")