#!/usr/bin/env python3
import os, glob, json, argparse, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, cv2, tempfile, shutil
from ultralytics import YOLO
from sklearn.metrics import r2_score
from pathlib import Path

# ---------- XML parsing ----------
def parse_xml_points(xml_path):
    pts=[]
    root = ET.parse(xml_path).getroot()
    for mt in root.findall(".//Marker_Type"):
        if (mt.findtext("Type") or "").strip() != "1":
            continue
        for m in mt.findall("Marker"):
            x = float(m.findtext("MarkerX")); y = float(m.findtext("MarkerY"))
            pts.append((x,y))
    return np.array(pts, dtype=np.float32)

# ---------- greedy one-to-one matching ----------
def match_points_greedy(gt_xy, det_xy, tol_px):
    if len(gt_xy)==0 and len(det_xy)==0: return 0,0,0,[]
    if len(gt_xy)==0: return 0, len(det_xy), 0, []
    if len(det_xy)==0: return 0, 0, len(gt_xy), []

    Ng, Nd = len(gt_xy), len(det_xy)
    D = np.sqrt(((gt_xy[:,None,:]-det_xy[None,:,:])**2).sum(axis=2))  # Ng x Nd
    matches=[]
    used_g = np.zeros(Ng, dtype=bool)
    used_d = np.zeros(Nd, dtype=bool)
    order = np.argsort(D, axis=None)
    for k in order:
        gi, di = np.unravel_index(k, D.shape)
        if used_g[gi] or used_d[di]: continue
        if D[gi, di] <= tol_px:
            matches.append((gi, di))
            used_g[gi] = True
            used_d[di] = True
    tp = len(matches)
    fp = int((~used_d).sum())
    fn = int((~used_g).sum())
    return tp, fp, fn, matches

# ---------- safe JSON load/dump (atomic writes) ----------
def safe_json_load(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        # backup the corrupt file and continue with empty cache
        try:
            shutil.copy2(path, path + ".corrupt")
            print(f"[roi] WARNING: unreadable ROI JSON, backed up -> {path+'.corrupt'}. Starting fresh.")
        except Exception:
            print(f"[roi] WARNING: unreadable ROI JSON. Starting fresh.")
        return {}

def safe_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="roi_", suffix=".json",
                               dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

# ---------- ROI picker ----------
_last_roi = None  # hotkey reuse buffer

def click_roi(img_bgr, title="Click top-left then bottom-right of 5×5 grid; press 'R' to reuse last"):
    global _last_roi
    clone = img_bgr.copy(); pts=[]
    def cb(ev,x,y,flags,param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x,y))
            cv2.circle(clone,(x,y),6,(0,0,255),-1)
            cv2.imshow(title, clone)
    cv2.imshow(title, clone)
    cv2.setMouseCallback(title, cb)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if len(pts) >= 2:
            break
        if k == 27:  # Esc cancels
            cv2.destroyWindow(title); return None
        if k in (ord('r'), ord('R')) and _last_roi:
            cv2.destroyWindow(title); return _last_roi
    cv2.destroyWindow(title)
    (x0,y0),(x1,y1) = pts
    x0,x1 = min(x0,x1), max(x0,x1)
    y0,y1 = min(y0,y1), max(y0,y1)
    _last_roi = (int(x0), int(y0), int(x1), int(y1))
    return _last_roi

def roi_is_valid(x0,y0,x1,y1,W,H):
    return (0 <= x0 < x1 <= W) and (0 <= y0 < y1 <= H) and (x1-x0 >= 10) and (y1-y0 >= 10)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def imread(p):
    img = cv2.imread(p)
    if img is None: raise FileNotFoundError(p)
    return img

def main(a):
    # model
    model = YOLO(a.weights)

    # ROI cache
    roi = safe_json_load(a.roi_json)

    # reclick set
    reclick_set = set()
    if a.reclick:
        for stem in a.reclick.split(","):
            stem = stem.strip()
            if stem: reclick_set.add(stem)

    # collect raw images
    exts = ("*.JPG","*.jpg","*.jpeg","*.PNG","*.png","*.tif","*.tiff")
    raws = []
    for e in exts:
        raws += glob.glob(os.path.join(a.in_dir, e))
    raws = sorted(raws)
    if not raws:
        print(f"[eval] No images found in {a.in_dir}")
        return

    out_dir = Path(a.out_dir) if a.out_dir else None
    if out_dir:
        ensure_dir(out_dir)
        dbg_dir = ensure_dir(out_dir / "overlays") if a.save_overlays else None

    rows = []

    for raw in raws:
        key = os.path.splitext(os.path.basename(raw))[0]
        xml_path = os.path.join(a.in_dir, f"CellCounter_{key}.xml")
        if not os.path.exists(xml_path):
            continue

        img = imread(raw); H,W = img.shape[:2]

        # Should we pick/re-pick ROI?
        need_click = False
        if a.reclick_all or (key in reclick_set) or (key not in roi):
            need_click = True
        else:
            x0,y0,x1,y1 = map(int, roi[key])
            if not roi_is_valid(x0,y0,x1,y1,W,H):
                need_click = True

        if need_click:
            print(f"[ROI] Choose ROI for {key} (press 'R' to reuse last; Esc cancels).")
            new_roi = click_roi(img, title=f"{key}: Click TL then BR of 5×5 grid (R=reuse last)")
            if new_roi is None:
                print(f"[skip] ROI not chosen for {key}")
                continue
            x0,y0,x1,y1 = new_roi
            roi[key] = [x0,y0,x1,y1]
            safe_json_dump(roi, a.roi_json)  # save immediately (atomic)
        else:
            x0,y0,x1,y1 = map(int, roi[key])

        # clamp for safety
        x0 = max(0, min(W-1, x0)); x1 = max(1, min(W, x1))
        y0 = max(0, min(H-1, y0)); y1 = max(1, min(H, y1))
        if not roi_is_valid(x0,y0,x1,y1,W,H):
            print(f"[warn] Bad ROI for {key}, skipping.")
            continue

        # ground truth inside ROI, convert to ROI-local coords
        gt_all = parse_xml_points(xml_path)
        if gt_all.size:
            m = (gt_all[:,0] >= x0) & (gt_all[:,0] < x1) & (gt_all[:,1] >= y0) & (gt_all[:,1] < y1)
            gt_roi_full = gt_all[m]
        else:
            gt_roi_full = np.empty((0,2), dtype=np.float32)

        gt_roi = gt_roi_full.copy()
        if gt_roi.size:
            gt_roi[:,0] -= x0
            gt_roi[:,1] -= y0

        # predict on ROI crop
        sub = img[y0:y1, x0:x1]
        pred = model.predict(
            source=sub,
            imgsz=a.imgsz,
            conf=a.conf,
            iou=a.iou,
            augment=a.tta,
            verbose=False
        )[0]

        det_xy = []
        if pred.boxes is not None:
            for (x1b, y1b, x2b, y2b, *_) in pred.boxes.xyxy.cpu().numpy():
                cx = (x1b + x2b) / 2.0
                cy = (y1b + y2b) / 2.0
                det_xy.append((cx, cy))
        det_xy = np.array(det_xy, dtype=np.float32) if len(det_xy) else np.empty((0,2), dtype=np.float32)

        # matching tolerance
        if a.match_um is not None:
            roi_w_px = (x1 - x0)              # 1 mm = 1000 um across the ROI width
            px_per_um = roi_w_px / 1000.0
            tol_px = float(a.match_um) * px_per_um
        else:
            tol_px = float(a.match_px)

        tp, fp, fn, matches = match_points_greedy(gt_roi, det_xy, tol_px)

        actual = int(len(gt_roi))
        pred_n = int(len(det_xy))
        diff = pred_n - actual
        abs_err = abs(diff)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        rows.append(dict(
            image=key,
            actual=actual, pred=pred_n,
            tp=tp, fp=fp, fn=fn,
            precision=prec, recall=rec, f1=f1,
            abs_err=abs_err, diff=diff,
            tol_px=tol_px
        ))

        if a.save_overlays and out_dir:
            ov = sub.copy()
            for (gx, gy) in gt_roi:
                cv2.drawMarker(ov, (int(gx), int(gy)), (255,0,0), cv2.MARKER_TILTED_CROSS, 18, 2)  # blue GT
            for (dx, dy) in det_xy:
                cv2.circle(ov, (int(dx), int(dy)), 6, (0,255,0), 2)                                 # green det
            for gi, di in matches:
                gx, gy = gt_roi[gi]; dx, dy = det_xy[di]
                cv2.line(ov, (int(gx),int(gy)), (int(dx),int(dy)), (0,255,255), 2)                  # yellow match
            txt = f"TP:{tp} FP:{fp} FN:{fn}  pred:{pred_n} actual:{actual}"
            cv2.putText(ov, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            out_path = out_dir / "overlays" / f"{key}_eval.png"
            cv2.imwrite(str(out_path), ov)

    if not rows:
        print("[eval] No images evaluated (check ROI/XML presence).")
        return

    df = pd.DataFrame(rows).sort_values("image")
    r2 = r2_score(df["actual"], df["pred"]) if df["actual"].sum()+df["pred"].sum() > 0 else float("nan")
    summary = {
        "n_images": int(len(df)),
        "sum_tp": int(df["tp"].sum()),
        "sum_fp": int(df["fp"].sum()),
        "sum_fn": int(df["fn"].sum()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "mean_f1": float(df["f1"].mean()),
        "mean_abs_err": float(df["abs_err"].mean()),
        "median_abs_err": float(df["abs_err"].median()),
        "r2_pred_vs_actual": float(r2),
        "tol_px_used": float(df["tol_px"].iloc[0])
    }

    out_csv = a.out_csv or "evaluate_spatial_results.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    sum_path = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(df[["image","actual","pred","tp","fp","fn","precision","recall","f1","abs_err"]])
    print("\nSummary:")
    for k,v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved per-image CSV → {out_csv}")
    print(f"Saved summary JSON   → {sum_path}")
    if a.save_overlays and out_dir:
        print(f"Overlays saved to    → {out_dir/'overlays'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Spatial evaluation vs CellCounter XML within ROI (with ROI re-picker)")
    ap.add_argument("--in_dir", required=True, help="Folder with RAW images + CellCounter_*.xml")
    ap.add_argument("--roi_json", required=True, help="ROI cache JSON (will be updated atomically)")
    ap.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    # matching tolerance: choose one
    ap.add_argument("--match_px", type=float, default=10.0, help="Tolerance radius in pixels")
    ap.add_argument("--match_um", type=float, default=None, help="Tolerance radius in micrometers (overrides --match_px)")
    # ROI re-pick options
    ap.add_argument("--reclick_all", action="store_true", help="Prompt ROI for every image")
    ap.add_argument("--reclick", default=None, help="Comma-separated image stems to re-pick (e.g., 'P8164336,P8164337')")
    ap.add_argument("--out_csv", default="evaluate_spatial_results.csv")
    ap.add_argument("--out_dir", default=None, help="Optional folder to save overlays")
    ap.add_argument("--save_overlays", action="store_true", help="Save TP/FP/FN overlays in out_dir/overlays")
    args = ap.parse_args()
    main(args)
