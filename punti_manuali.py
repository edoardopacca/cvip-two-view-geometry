import cv2
import numpy as np
import os

IMG1_PATH = "object/img1.png"
IMG2_PATH = "object/img2.png"
OUT_DIR = "outputs"
OUT_FILE = os.path.join(OUT_DIR, "manual_high_precision.npz")

os.makedirs(OUT_DIR, exist_ok=True)

img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)
assert img1 is not None and img2 is not None, "Errore caricamento immagini."

h1, w1 = img1.shape[:2]
combo_orig = cv2.hconcat([img1, img2])
combo_disp = combo_orig.copy()

pts1 = []
pts2 = []
expecting = 1

cursor_x, cursor_y = w1 // 2, h1 // 2
zoom_factor = 6
patch_radius = 30

def redraw():
    global combo_disp
    combo_disp = combo_orig.copy()
    for i, pt in enumerate(pts1):
        cv2.circle(combo_disp, pt, 4, (0, 0, 255), -1)
        cv2.putText(combo_disp, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    for i, pt in enumerate(pts2):
        pt_disp = (int(pt[0] + w1), int(pt[1]))
        cv2.circle(combo_disp, pt_disp, 4, (0, 255, 0), -1)
        cv2.putText(combo_disp, str(i+1), (pt_disp[0]+8, pt_disp[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if i < len(pts1):
            cv2.line(combo_disp, pts1[i], pt_disp, (255, 255, 0), 1)

    msg = "1. FOTO SINISTRA" if expecting == 1 else "2. FOTO DESTRA"
    color = (0, 0, 255) if expecting == 1 else (0, 255, 0)
    cv2.putText(combo_disp, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

def save_point():
    global expecting
    x, y = cursor_x, cursor_y
    if expecting == 1 and x < w1:
        pts1.append((x, y))
        expecting = 2
        print(f"✅ Punto {len(pts1)} registrato su SINISTRA.")
    elif expecting == 2 and x >= w1:
        pts2.append((x - w1, y))
        expecting = 1
        print(f"✅ Punto {len(pts2)} registrato su DESTRA.")
    else:
        print("⚠️ Cursore fuori posto! Spostalo nell'immagine corretta.")
    redraw()

def update_lens():
    ch, cw = combo_orig.shape[:2]
    y_min, y_max = max(0, cursor_y - patch_radius), min(ch, cursor_y + patch_radius)
    x_min, x_max = max(0, cursor_x - patch_radius), min(cw, cursor_x + patch_radius)
    patch = combo_orig[y_min:y_max, x_min:x_max].copy()
    if patch.shape[0] > 0 and patch.shape[1] > 0:
        zoomed = cv2.resize(patch, (0,0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
        zh, zw = zoomed.shape[:2]
        cv2.line(zoomed, (zw//2, 0), (zw//2, zh), (255, 0, 0), 1)
        cv2.line(zoomed, (0, zh//2), (zw, zh//2), (255, 0, 0), 1)
        cv2.imshow("Lente", zoomed)

def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        save_point()

cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main", 1600, 600)
cv2.namedWindow("Lente", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Main", mouse_callback)

redraw()

try:
    while True:
        temp_disp = combo_disp.copy()
        cv2.drawMarker(temp_disp, (cursor_x, cursor_y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        update_lens()
        cv2.imshow("Main", temp_disp)

        key = cv2.waitKeyEx(15)
        if key == -1:
            continue
        elif key == ord('q') or key == 27:
            break
        elif key == ord('z'):
            if expecting == 2:
                pts1.pop()
                expecting = 1
                print("Undo: Punto Sinistra rimosso.")
            elif expecting == 1 and len(pts2) > 0:
                pts2.pop()
                expecting = 2
                print("Undo: Punto Destra rimosso.")
            redraw()
        elif key == ord(' '):
            save_point()
        elif key == ord('w') or key in [63232, 2490368, 82]: cursor_y -= 1
        elif key == ord('s') or key in [63233, 2621440, 84]: cursor_y += 1
        elif key == ord('a') or key in [63234, 2424832, 81]: cursor_x -= 1
        elif key == ord('d') or key in [63235, 2555904, 83]: cursor_x += 1
        cursor_x = max(0, min(cursor_x, combo_orig.shape[1] - 1))
        cursor_y = max(0, min(cursor_y, combo_orig.shape[0] - 1))

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()

if len(pts1) == len(pts2) and len(pts1) > 0:
    np.savez(OUT_FILE, pts1=np.array(pts1, dtype=np.float32), pts2=np.array(pts2, dtype=np.float32))
    print(f"\n🎉 Salvati {len(pts1)} punti in: {OUT_FILE}")
else:
    print("\n⚠️ Numero punti dispari o zero. Nessun salvataggio.")