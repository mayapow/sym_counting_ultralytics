#!/usr/bin/env python3
import os, glob, json, argparse, cv2, pandas as pd
from ultralytics import YOLO

# ---------- small utils ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def stem(p): return os.path.splitext(os.path.basename(p))[0]

def imread(p, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(p, flags)
    if img is None:
        raise FileNotFoundError(p)
    return img

def click_roi(img_bgr, title="Click TL then BR of 5×5 area"):
    """Manual ROI picker: click top-left then bottom-right. Press Esc to cancel."""
    clone = img_bgr.copy(); pts=[]
    def cb(ev,x,y,flags,param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x,y))
            cv2.circle(clone,(x,y),6,(0,0,255),-1)
            cv2.imshow(title, clone)
    cv2.imshow(title, clone)
    cv2.setMouseCallback(title, cb)
    while len(pts) < 2:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow(title)
    if len(pts) < 2:
        return None
    (x0,y0),(x1,y1) = pts
    x0,x1 = min(x0,x1), max(x0,x1)
    y0,y1 = min(y0,y1), max(y0,y1)
    return int(x0), int(y0), int(x1), int(y1)

# ---------- main ----------
def main(args):
    ensure_dir(args.out_dir)
    model = YOLO(args.weights)

    # load / init ROI cache
    roi = {}
    if args.roi_json and os.path.exists(args.roi_json):
        roi = json.load(open(args.roi_json))

    # find raw images
    raws = sorted(glob.glob(os.path.join(args.in_dir, "**", "*.JPG"), recursive=True))
    raws += sorted(glob.glob(os.path.join(args.in_dir, "**", "*.jpg"), recursive=True))
    raws += sorted(glob.glob(os.path.join(args.in_dir, "**", "*.jpeg"), recursive=True))
    if not raws:
        print("[infer] no images found in --in_dir"); return

    rows=[]
    for raw in raws:
        name = os.path.basename(raw)
        key  = stem(raw)
        img  = imread(raw); H,W = img.shape[:2]

        # get or create ROI
        if key in roi:
            x0,y0,x1,y1 = roi[key]
        else:
            print(f"[ROI] No ROI for {key}. Please click TL then BR for the 5×5 area.")
            clicked = click_roi(img, title=f"{key}: Click TL then BR (Esc to cancel)")
            if clicked is None:
                print(f"[skip] ROI not chosen for {key}")
                continue
            x0,y0,x1,y1 = clicked
            roi[key] = [x0,y0,x1,y1]
            if args.roi_json:
                os.makedirs(os.path.dirname(args.roi_json), exist_ok=True)
                json.dump(roi, open(args.roi_json,"w"), indent=2)

        # crop to ROI and run detection
        sub = img[y0:y1, x0:x1]
        res = model.predict(
            source=sub,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            augment=args.tta,       # TTA if flag is set
            verbose=False
        )[0]

        # gather detections
        det = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        count = len(det)

        # draw overlay
        overlay = img.copy()
        # ROI rectangle (red)
        cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,0,255),3)
        # boxes (green) drawn in full-image coords
        for (x1b,y1b,x2b,y2b,*_) in det:
            cv2.rectangle(
                overlay,
                (x0 + int(x1b), y0 + int(y1b)),
                (x0 + int(x2b), y0 + int(y2b)),
                (0,255,0), 2
            )
        cv2.putText(overlay, f"Count: {count}", (x0+10, y0+40), 0, 1, (0,0,255), 2)

        out_png = os.path.join(args.out_dir, f"{key}_det.png")
        cv2.imwrite(out_png, overlay)

        rows.append(dict(image_id=key, count=count, overlay=out_png))

    # write counts table
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "counts.csv"), index=False)
    print(f"[done] overlays + counts.csv -> {args.out_dir}")
    if args.roi_json:
        print(f"[roi] cache saved -> {args.roi_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("ROI-aware YOLO inference + counting (prompts for missing ROIs)")
    ap.add_argument("--in_dir", required=True, help="Folder of images to count")
    ap.add_argument("--out_dir", required=True, help="Output folder for overlays and counts.csv")
    ap.add_argument("--roi_json", required=True, help="ROI cache JSON (will be created/updated)")
    ap.add_argument("--weights", required=True, help="Path to best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--tta",  action="store_true", help="Enable test-time augmentation")
    args = ap.parse_args()
    main(args)
