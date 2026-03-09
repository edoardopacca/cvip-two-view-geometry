# Homework IPCV — Two-view geometry

Everything related to code, reproducibility, and outputs is inside the **`code_and_outputs/`** folder. In particular, the **notebook** (`code_and_outputs/notebook.ipynb`) contains all the code used for the homework (calibration, eight-point algorithm, epipolar lines, triangulation, 3D visualization). The only separate Python file from the notebook is **`code_and_outputs/punti_manuali.py`**: it is used to annotate manual points and works correctly only when run from the terminal.

The report is in the PDF **`Report_homework_Ferrari_Paccagnella_Errico.pdf`**.

**`3d_view.html`** is the reconstructed view of the structure.

**`Images_used_in_the_report/`** — Subset of images from `outputs/` that were used in the report.

---

## Folder structure

Inside **`code_and_outputs/`**:

- **`object/`** — Photos of the 3D structure we had to reconstruct (scene with the letters).
- **`checkerboard/`** — Checkerboard photos used for calibration (different poses).
- **`manual_points/`** — `.npz` files with the annotated manual points (correspondences for letters/parts of the structure).
- **`debug_corners/`** — Calibration output images (e.g. corners detected on the checkerboard).
- **`camera_calib.npz`** — Camera intrinsic parameters and distortion coefficients saved after calibration. Used by the notebook and by `punti_manuali.py`.
- **`outputs/`** — Outputs from the various pipeline steps.

### Contents of `code_and_outputs/outputs/`

- **`good_matches.png`** — Visual output of SIFT correspondences (matches between the two images).
- **`8point_epilines_img1_*.png`** and **`8point_epilines_img2_*.png`** — Epipolar lines on image 1 and image 2 computed with our eight-point algorithm; there are versions with different point sizes (`r2`, `r16`) and different subsets of matches (`all`, `rand25m25s`).
- **`epilines_img1_opencv5pt.png`** — Epipolar lines on image 1 computed with the OpenCV five-point algorithm.
- **`img1_undistorted.png`**, **`img2_undistorted.png`** — The two scene images after distortion correction (undistort).
- **`matches_px.npz`** — Corresponding points in pixels (used for essential matrix estimation and triangulation).
- **`E_results.npz`** — Results related to the essential matrix (and/or poses) saved for later use.
- **`3d_view.html`** — 3D visualization of the reconstructed scene (open in a browser). A copy is also in the project root, since it is particularly important.
