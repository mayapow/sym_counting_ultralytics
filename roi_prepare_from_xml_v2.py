#!/usr/bin/env python3
import os, glob, json, random, argparse, xml.etree.ElementTree as ET
from pathlib import Path
import cv2

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def stem(p): return os.path.splitext(os.path.basename(p))[0]

def imread(p, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(p, flags)
    if img is None:
        raise FileNotFoundError(p)
    return img

def parse_xml_points(xml_path):
    """Return list of (x,y) for CellCounter Type==1 markers."""
    pts=[]
    root = ET.parse(xml_path).getroot()
    # Accept both Marker_Type/Type or Marker_Type/@type styles
    for mtype in root.findall(".//Marker_Type"):
        t = (mtype.findtext("Type") or "").strip()
        if t != "1":
            continue
        for m in mtype.findall("Marker"):
            try:
                x = int(float(m.findtext("MarkerX")))
                y = int(float(m.findtext("MarkerY")))
                pts.append((x,y))
            except Exception:
                pass
    return pts

def click_roi(img_bgr, title="Click TL then BR of 5×5 area"):
    clone = img_bgr.copy(); pts=[]
    def cb(ev,x,y,flags,param):
        if ev==cv2.EVENT_LBUTTONDOWN and len(pts)<2:
            pts.append((x,y))
            cv2.circle(clone,(x,y),6,(0,0,255),-1)
            cv2.imshow(title, clone)
    cv2.imshow(title, clone)
    cv2.setMouseCallback(title, cb)
    while len(pts)<2:
        if cv2.waitKey(1) & 0xFF == 27: break
    cv2.destroyWindow(title)
    if len(pts)<2: return None
    (x0,y0),(x1,y1)=pts
    x0,x1=min(x0,x1),max(x0,x1); y0,y1=min(y0,y1),max(y0,y1)
    return int(x0),int(y0),int(x1),int(y1)

def main(args):
    random.seed(0)
    out = Path(args.out_dir)
    img_tr = ensure_dir(out/"images/train")
    img_va = ensure_dir(out/"images/val")
    lbl_tr = ensure_dir(out/"labels/train")
    lbl_va = ensure_dir(out/"labels/val")
    ovl_dir = ensure_dir(out/"overlays")

    # ROI cache
    roi = {}
    if args.roi_json and os.path.exists(args.roi_json):
        roi = json.load(open(args.roi_json))

    # collect raws
    raws = sorted(glob.glob(os.path.join(args.in_dir, "**", "*.JPG"), recursive=True))
    raws += sorted(glob.glob(os.path.join(args.in_dir, "**", "*.jpg"), recursive=True))
    raws += sorted(glob.glob(os.path.join(args.in_dir, "**", "*.jpeg"), recursive=True))
    if not raws:
        print("[prep] No raw images found. Check --in_dir.")
        return
        
    # Build a stable list of sample keys (one per RAW)
    samples = [os.path.splitext(os.path.basename(p))[0] for p in raws]
    
    # Shuffle once (deterministic)
    random.seed(0)
    idx = list(range(len(samples)))
    random.shuffle(idx)
    
    # Determine number of validation images
    n_val = max(1, int(round(args.val_frac * len(samples))))
    val_idx = set(idx[:n_val])
    
    # Map image → train/val
    key_to_subset = {samples[i]: ("val" if i in val_idx else "train")
                 for i in range(len(samples))}

    # CSV report
    report_rows = []

    kept = 0
    for raw in raws:
        key = stem(raw)
        # match CellCounter XML
        xml_candidates = []
        for pat in [f"CellCounter_{key}.xml", f"cellcounter_{key}.xml", f"{key}.xml"]:
            xml_candidates += glob.glob(os.path.join(args.in_dir, "**", pat), recursive=True)
        if not xml_candidates:
            # still write image as negative-only (no labels), but warn
            xml_path = None
        else:
            xml_path = sorted(xml_candidates)[0]

        img = imread(raw); H,W = img.shape[:2]

        # ROI
        if key in roi:
            x0,y0,x1,y1 = roi[key]
        else:
            click = click_roi(img, title=f"{key}: Click TL then BR of 5×5 (Esc to cancel)")
            if click is None:
                print(f"[skip] no ROI clicked for {key}")
                report_rows.append(dict(image=key, has_xml=bool(xml_path), pts_total=0, pts_used=0, note="no ROI"))
                continue
            x0,y0,x1,y1 = click
            roi[key] = [x0,y0,x1,y1]
            if args.roi_json:
                os.makedirs(os.path.dirname(args.roi_json), exist_ok=True)
                json.dump(roi, open(args.roi_json,"w"), indent=2)

        ppm = max(1.0, (x1-x0)/1.0)  # pixels per mm; guard divide-by-zero
        box_px = max(4, int(round((args.box_um/1000.0)*ppm)))

        pts = []
        if xml_path:
            pts = parse_xml_points(xml_path)
        pts_total = len(pts)

        # optionally filter to ROI
        if args.filter_to_roi:
            pts_roi = [(x,y) for (x,y) in pts if (x0<=x<x1 and y0<=y<y1)]
        else:
            pts_roi = pts[:]

        # ALWAYS write image and label file (even if zero labels)
        subset = key_to_subset[key]
        out_img = (out/f"images/{subset}"/os.path.basename(raw)).as_posix()
        out_lbl = (out/f"labels/{subset}"/f"{key}.txt").as_posix()

        cv2.imwrite(out_img, img)

        yolo_lines=[]
        for (cx,cy) in pts_roi:
            w=h=box_px
            x1b=max(0,min(W-1,cx-w//2)); y1b=max(0,min(H-1,cy-h//2))
            x2b=max(0,min(W-1,x1b+w));   y2b=max(0,min(H-1,y1b+h))
            bx=((x1b+x2b)/2)/W; by=((y1b+y2b)/2)/H
            bw=(x2b-x1b)/W;     bh=(y2b-y1b)/H
            yolo_lines.append(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
        with open(out_lbl,"w") as f:
            f.write("\n".join(yolo_lines))

        # QC overlay: red ROI + blue crosses (only points INSIDE ROI are drawn)
        ov = img.copy()
        cv2.rectangle(ov,(x0,y0),(x1,y1),(0,0,255),3)
        for (cx,cy) in [(x,y) for (x,y) in pts if (x0<=x<x1 and y0<=y<y1)]:
            cv2.drawMarker(ov,(cx,cy),(255,0,0),cv2.MARKER_TILTED_CROSS,16,2)
        cv2.imwrite(os.path.join(ovl_dir, f"{key}_truth.png"), ov)

        report_rows.append(dict(
            image=key, has_xml=bool(xml_path), pts_total=pts_total,
            pts_used=len(pts_roi), subset=subset, out_img=out_img, out_lbl=out_lbl
        ))
        kept += 1

    # dataset YAML
    with open(out/"sym_dataset.yaml","w") as f:
        f.write(f"""path: {out.as_posix()}
train: images/train
val: images/val
names:
  0: symbiont
""")

    # write report
    import pandas as pd
    pd.DataFrame(report_rows).to_csv(out/"prep_report.csv", index=False)

    print(f"[done] wrote {kept} images to images/{{train,val}} and labels/{{train,val}}")
    print(f"[yaml] {out/'sym_dataset.yaml'}")
    print(f"[report] {out/'prep_report.csv'}")
    if args.roi_json: print(f"[roi] cache -> {args.roi_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("YOLO prep from CellCounter XML + manual ROI (robust)")
    ap.add_argument("--in_dir", required=True, help="Folder with RAW + CellCounter_*.xml")
    ap.add_argument("--out_dir", required=True, help="Output dataset folder")
    ap.add_argument("--roi_json", required=True, help="Cache of ROI clicks (JSON)")
    ap.add_argument("--box_um", type=float, default=12.0, help="Box side (µm)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--filter_to_roi", type=lambda s:s.lower() in ["1","true","t","yes","y"], default=True,
                    help="If true, keep only labels inside ROI (default true). Set false to include all labels.")
    args = ap.parse_args()
    main(args)
