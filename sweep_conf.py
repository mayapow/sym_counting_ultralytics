#!/usr/bin/env python3
import os, glob, json, argparse, xml.etree.ElementTree as ET
import numpy as np, cv2, pandas as pd
from ultralytics import YOLO

def parse_xml_points(xml_path):
    pts=[]
    root = ET.parse(xml_path).getroot()
    for mt in root.findall(".//Marker_Type"):
        if (mt.findtext("Type") or "").strip() != "1": continue
        for m in mt.findall("Marker"):
            x = int(float(m.findtext("MarkerX"))); y = int(float(m.findtext("MarkerY")))
            pts.append((x,y))
    return np.array(pts, dtype=np.float32)

def match_counts(gt_xy, det_xy, match_radius_px=10):
    if len(gt_xy)==0: 
        return 0, len(det_xy), 0
    if len(det_xy)==0:
        return 0, 0, len(gt_xy)
    used = np.zeros(len(det_xy), dtype=bool)
    tp=0
    for g in gt_xy:
        d2 = np.sum((det_xy - g)**2, axis=1)
        j = np.argmin(d2)
        if not used[j] and d2[j] <= match_radius_px**2:
            used[j]=True; tp+=1
    fp = int((~used).sum())
    fn = int(len(gt_xy) - tp)
    return tp, fp, fn

def main(a):
    model = YOLO(a.weights)
    roi = json.load(open(a.roi_json))
    raws = sorted(glob.glob(os.path.join(a.in_dir, "**", "*.JPG"), recursive=True))
    raws += sorted(glob.glob(os.path.join(a.in_dir, "**", "*.jpg"), recursive=True))
    rows=[]
    for conf in a.confs:
        tp=fp=fn=0; mae=[]
        for raw in raws:
            key = os.path.splitext(os.path.basename(raw))[0]
            # ground truth
            xml = glob.glob(os.path.join(a.in_dir, "**", f"CellCounter_{key}.xml"), recursive=True)
            if not xml: 
                continue
            gt = parse_xml_points(xml[0])
            if key not in roi: 
                continue
            x0,y0,x1,y1 = roi[key]
            gt_roi = gt[(gt[:,0]>=x0)&(gt[:,0]<x1)&(gt[:,1]>=y0)&(gt[:,1]<y1)]
            # predict within ROI
            img = cv2.imread(raw)
            sub = img[y0:y1, x0:x1]
            res = model.predict(source=sub, imgsz=a.imgsz, conf=conf, iou=a.iou, augment=a.tta, verbose=False)[0]
            det = []
            if res.boxes is not None:
                for b in res.boxes.xyxy.cpu().numpy():
                    x1b,y1b,x2b,y2b = b[:4]
                    det.append(((x1b+x2b)/2.0, (y1b+y2b)/2.0))
            det = np.array(det, dtype=np.float32)
            # match on centers with a 1/2 cell diameter tolerance
            tp_i, fp_i, fn_i = match_counts(gt_roi, det, match_radius_px=a.match_px)
            tp+=tp_i; fp+=fp_i; fn+=fn_i
            mae.append(abs(len(det) - len(gt_roi)))
        prec = tp/(tp+fp+1e-9)
        rec  = tp/(tp+fn+1e-9)
        f1   = 2*prec*rec/(prec+rec+1e-9)
        rows.append(dict(conf=conf, precision=prec, recall=rec, f1=f1, MAE=np.mean(mae) if mae else np.nan))
    df = pd.DataFrame(rows).sort_values("conf")
    print(df.to_string(index=False))
    if a.out_csv:
        df.to_csv(a.out_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Confidence sweep with ROI + XML truth")
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--roi_json", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--confs", type=float, nargs="+", default=[0.25,0.3,0.35,0.4,0.45,0.5])
    ap.add_argument("--match_px", type=int, default=10, help="Distance (px) to accept a predicted center as a match")
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()
    main(a)
