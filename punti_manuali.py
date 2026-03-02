import cv2
import numpy as np
import os
from datetime import datetime

# ============================================================
#  MANUAL POINTS TOOL (UNDISTORT MODE)
#  - Mostra img1/img2 UNDISTORTE (cv2.undistort con K,dist)
#  - I punti salvati (pts1/pts2) sono PIXEL UNDISTORTI
#  - La linea epipolare usa F_cv (o F_8) da E_results.npz (coerente con undistort)
#
#  ⚠️ NOTA per il tuo Step 3:
#  Se usi questo tool, NON fare undistortPoints(..., K, dist) sui punti manuali
#  perché sono già undistorti. Usa:
#     m_p1_n = cv2.undistortPoints(m_p1.reshape(-1,1,2), K, None).reshape(-1,2)
#  oppure:
#     m_p1_n = cv2.undistortPoints(m_p1.reshape(-1,1,2), K, np.zeros((1,5))).reshape(-1,2)
# ============================================================

# ------------------ PATHS ------------------
IMG1_PATH = "object/img1.JPG"
IMG2_PATH = "object/img2.JPG"
CALIB_PATH = "camera_calib.npz"

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

E_RESULTS_PATH = os.path.join(OUT_DIR, "E_results.npz")
MATCHES_F_PATH = os.path.join(OUT_DIR, "matches_inliers_px.npz")  # fallback (di solito distorto)

# Ogni avvio = una sessione diversa (file diversi)
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_FILE = os.path.join(OUT_DIR, f"manual_session_{SESSION_ID}.npz")
FINAL_FILE   = os.path.join(OUT_DIR, f"manual_high_precision_{SESSION_ID}.npz")

print(f"🆔 Sessione: {SESSION_ID}")
print(f"💾 Autosave sessione: {SESSION_FILE}")
print(f"🏁 File finale (se punti pari): {FINAL_FILE}")

# ------------------ LOAD CALIB ------------------
cal = np.load(CALIB_PATH)
K = cal["K"].astype(np.float64)
dist = cal["dist"].astype(np.float64)

# ------------------ LOAD IMAGES (UNDISTORT) ------------------
img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)
assert img1 is not None and img2 is not None, "Errore caricamento immagini."

# Undistort con la stessa K (coerente col tuo Step2)
img1_u = cv2.undistort(img1, K, dist)
img2_u = cv2.undistort(img2, K, dist)

h1, w1 = img1_u.shape[:2]
h2, w2 = img2_u.shape[:2]
assert h1 == h2, "Le immagini devono avere la stessa altezza per hconcat."

combo_orig = cv2.hconcat([img1_u, img2_u])
combo_disp = combo_orig.copy()

# ------------------ LOAD FUNDAMENTAL MATRIX F (UNDISTORT) ------------------
def load_F():
    F = None

    # Preferisci E_results: F_cv o F_8 (coerenti con undistort)
    if os.path.exists(E_RESULTS_PATH):
        d = np.load(E_RESULTS_PATH)
        if "F_cv" in d:
            F = d["F_cv"].astype(np.float64)
            print(f"✅ Caricata F da {E_RESULTS_PATH} (F_cv) [UNDISTORT]")
        elif "F_8" in d:
            F = d["F_8"].astype(np.float64)
            print(f"✅ Caricata F da {E_RESULTS_PATH} (F_8) [UNDISTORT]")

    # Fallback: matches_inliers_px.npz (spesso è per DISTORTO → warning)
    if F is None and os.path.exists(MATCHES_F_PATH):
        d = np.load(MATCHES_F_PATH)
        if "F" in d:
            F = d["F"].astype(np.float64)
            print(f"⚠️ Caricata F da {MATCHES_F_PATH} (F) [PROBABILE DISTORTO → potrebbe NON combaciare con UNDISTORT]")

    return F

F = load_F()
if F is None:
    print("⚠️ ATTENZIONE: Non ho trovato nessuna matrice fondamentale F.")
    print("   Crea prima outputs/E_results.npz (consigliato) e rilancia.")
else:
    # normalizza scala (opzionale)
    if abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]

# ------------------ STATE ------------------
pts1 = []
pts2 = []
expecting = 1  # 1 = click sinistra, 2 = click destra

cursor_x, cursor_y = w1 // 2, h1 // 2
zoom_factor = 6
patch_radius = 30

current_epi_line_right = None  # (a,b,c) sulla FOTO DESTRA (pixel undistorti)

# ------------------ SAVE HELPERS ------------------
def atomic_save_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)

def save_session():
    atomic_save_npz(
        SESSION_FILE,
        pts1=np.array(pts1, dtype=np.float32),
        pts2=np.array(pts2, dtype=np.float32),
        expecting=np.array([expecting], dtype=np.int32),
        cursor=np.array([cursor_x, cursor_y], dtype=np.int32),
        w1=np.array([w1], dtype=np.int32),
        img1=np.array([IMG1_PATH]),
        img2=np.array([IMG2_PATH]),
        session_id=np.array([SESSION_ID]),
        undistorted=np.array([1], dtype=np.int32),  # flag utile
        K=K,
    )

def save_final_if_possible():
    if len(pts1) == len(pts2) and len(pts1) > 0:
        np.savez(
            FINAL_FILE,
            pts1=np.array(pts1, dtype=np.float32),
            pts2=np.array(pts2, dtype=np.float32),
            img1=np.array([IMG1_PATH]),
            img2=np.array([IMG2_PATH]),
            session_id=np.array([SESSION_ID]),
            undistorted=np.array([1], dtype=np.int32),
            K=K,
        )
        print(f"🎉 Salvato file finale: {FINAL_FILE}  ({len(pts1)} punti)")
        return True
    else:
        print("⚠️ Non posso salvare finale: punti dispari o zero.")
        return False

# ------------------ EPIPOLAR UTILITIES ------------------
def compute_epiline_right_from_left(pt_left):
    """pt_left: (u,v) pixel UNDISTORTI su foto sinistra. Ritorna linea su destra ax+by+c=0."""
    if F is None:
        return None
    x1 = np.array([pt_left[0], pt_left[1], 1.0], dtype=np.float64)
    l2 = F @ x1
    a, b, c = l2
    n = np.sqrt(a * a + b * b) + 1e-12
    return np.array([a / n, b / n, c / n], dtype=np.float64)

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

def draw_epiline_on_right(vis, line_right, color=(0, 255, 255), thickness=2):
    """Disegna la retta epipolare nella metà destra (offset w1)."""
    if line_right is None:
        return
    a, b, c = line_right
    p0, p1 = line_segment_in_image(a, b, c, w2, h2)
    if p0 is None:
        return
    p0d = (p0[0] + w1, p0[1])
    p1d = (p1[0] + w1, p1[1])
    cv2.line(vis, p0d, p1d, color, thickness)

def cursor_dist_to_line_right(line_right, cx_combo, cy_combo):
    if line_right is None or cx_combo < w1:
        return None
    x = float(cx_combo - w1)
    y = float(cy_combo)
    a, b, c = line_right
    return abs(a * x + b * y + c)

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

    # linea epipolare sulla destra quando stai scegliendo il punto dx
    if expecting == 2 and current_epi_line_right is not None:
        draw_epiline_on_right(combo_disp, current_epi_line_right, color=(0, 255, 255), thickness=2)

    # testi UI
    msg = "1. FOTO SINISTRA (UNDISTORT)" if expecting == 1 else "2. FOTO DESTRA (UNDISTORT)"
    color = (0, 0, 255) if expecting == 1 else (0, 255, 0)
    cv2.putText(combo_disp, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    info = f"SX:{len(pts1)}  DX:{len(pts2)}  | click: {'SX' if expecting==1 else 'DX'}"
    cv2.putText(combo_disp, info, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if F is None:
        cv2.putText(combo_disp, "F non trovata: epipolar line OFF", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# --------- LENTE: usa il frame mostrato (temp_disp) ----------
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
        # crocino BLU dentro la lente (quello rimane)
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
        print(f"✅ Punto {len(pts1)} registrato su SINISTRA (UNDISTORT). (epiline -> DESTRA)")
    elif expecting == 2 and x >= w1:
        pts2.append((x - w1, y))
        expecting = 1
        update_current_epiline()
        print(f"✅ Punto {len(pts2)} registrato su DESTRA (UNDISTORT).")
    else:
        print("⚠️ Cursore fuori posto! Spostalo nell'immagine corretta.")

    redraw()
    save_session()

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
save_session()

print("\nControlli:")
print("  - Click sinistro: salva punto (alternando SX -> DX)")
print("  - Spazio: salva punto (come click)")
print("  - WASD / frecce: muovi cursore di 1px")
print("  - z: undo ultimo punto (gestisce SX/DX)")
print("  - f: salva file finale (solo se punti pari)")
print("  - q o ESC: esci (autosave sempre, finale solo se punti pari)\n")

# ------------------ MAIN LOOP ------------------
try:
    while True:
        temp_disp = combo_disp.copy()

        # (RIMOSSO) niente croce gialla sul main

        # distanza cursore-linea (utile quando stai scegliendo a destra)
        if expecting == 2 and current_epi_line_right is not None:
            d = cursor_dist_to_line_right(current_epi_line_right, cursor_x, cursor_y)
            if d is not None:
                cv2.putText(temp_disp, f"dist to epiline: {d:.2f}px", (50, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        update_lens(temp_disp)
        cv2.imshow("Main", temp_disp)

        key = cv2.waitKeyEx(15)
        if key == -1:
            continue

        # exit
        if key == ord('q') or key == 27:
            break

        # undo
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
            save_session()

        # save point
        elif key == ord(' '):
            save_point()

        # save final
        elif key == ord('f'):
            save_session()
            save_final_if_possible()

        # movement keys (WASD + frecce)
        elif key == ord('w') or key in [63232, 2490368, 82]:
            cursor_y -= 1
        elif key == ord('s') or key in [63233, 2621440, 84]:
            cursor_y += 1
        elif key == ord('a') or key in [63234, 2424832, 81]:
            cursor_x -= 1
        elif key == ord('d') or key in [63235, 2555904, 83]:
            cursor_x += 1

        # clamp cursor
        cursor_x = max(0, min(cursor_x, combo_orig.shape[1] - 1))
        cursor_y = max(0, min(cursor_y, combo_orig.shape[0] - 1))

except KeyboardInterrupt:
    pass

# ------------------ CLEANUP + SAVE ------------------
cv2.destroyAllWindows()

save_session()
print(f"\n💾 Sessione salvata in: {SESSION_FILE}")

if len(pts1) == len(pts2) and len(pts1) > 0:
    np.savez(
        FINAL_FILE,
        pts1=np.array(pts1, dtype=np.float32),
        pts2=np.array(pts2, dtype=np.float32),
        undistorted=np.array([1], dtype=np.int32),
        K=K,
    )
    print(f"🎉 Salvato file finale: {FINAL_FILE}")
else:
    print("⚠️ Nessun salvataggio finale (punti dispari o zero).")