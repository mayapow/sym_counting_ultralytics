#!/usr/bin/env python3
import os, glob, json, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, cv2
from ultralytics import YOLO
from sklearn.metrics import r2_score
import argparse

def parse_xml_points(xml_path):
    """Extract (x,y) coordinates from Fiji CellCounter XML file."""
    pts=[]
    root = ET.parse(xml_path).getroot()
    for mt in root.findall(".//Marker_Type"):
        if (mt.findtext("Type") or "").strip() != "1":
            continue
        for m in mt.findall("Marker"):
            x=float(m.findtext("MarkerX")); y=float(m.findtext("MarkerY"))
            pts.append((x,y))
    return np.array(pts, dtype=np.float32)

def match_counts(gt_xy, det_xy, tol_px=10):
    """Return TP, FP, FN between GT and predicted points."""
    if len(gt_xy)==0: return 0, len(det_xy), 0
    if len(det_xy)==0: return 0, 0, len(gt_xy)
    used=np.zeros(len(det_xy),dtype=bool)
    tp=0
    for g in gt_xy:
        d2=np.sum((det_xy-g)**2,axis=1)
        j=np.argmin(d2)
        if not used[j] and d2[j]<=tol_px**2:
            used[j]=True; tp+=1
    fp=int((~used).sum()); fn=int(len(gt_xy)-tp)
    return tp,fp,fn

def main(args):
    model=YOLO(args.weights)
    roi=json.load(open(args.roi_json))
    raws=sorted(glob.glob(os.path.join(args.in_dir,"*.JPG")))
    records=[]
    for raw in raws:
        key=os.path.splitext(os.path.basename(raw))[0]
        xml_path=os.path.join(args.in_dir,f"CellCounter_{key}.xml")
        if not os.path.exists(xml_path) or key not in roi:
            continue
        x0,y0,x1,y1=roi[key]
        img=cv2.imread(raw)
        sub=img[y0:y1,x0:x1]
        gt=parse_xml_points(xml_path)
        gt_roi=gt[(gt[:,0]>=x0)&(gt[:,0]<x1)&(gt[:,1]>=y0)&(gt[:,1]<y1)]

        res=model.predict(source=sub,imgsz=args.imgsz,conf=args.conf,iou=args.iou,verbose=False)[0]
        det=[]
        if res.boxes is not None:
            for b in res.boxes.xyxy.cpu().numpy():
                x1b,y1b,x2b,y2b=b[:4]
                det.append(((x1b+x2b)/2.0,(y1b+y2b)/2.0))
        det=np.array(det,dtype=np.float32)

        tp,fp,fn=match_counts(gt_roi,det,args.match_px)
        records.append(dict(image=key,tp=tp,fp=fp,fn=fn,
                            actual=len(gt_roi),pred=len(det)))

    df=pd.DataFrame(records)
    df["precision"]=df.tp/(df.tp+df.fp+1e-9)
    df["recall"]=df.tp/(df.tp+df.fn+1e-9)
    df["f1"]=2*df.precision*df.recall/(df.precision+df.recall+1e-9)
    df["abs_err"]=(df.pred-df.actual).abs()
    df["rel_err"]=(df.pred-df.actual)/df.actual

    # --- summary printout ---
    print("\nPer-image metrics:")
    print(df[["image","actual","pred","tp","fp","fn","precision","recall","f1","abs_err"]])

    print("\nSummary:")
    print(f"Mean Precision: {df.precision.mean():.3f}")
    print(f"Mean Recall:    {df.recall.mean():.3f}")
    print(f"Mean F1:        {df.f1.mean():.3f}")
    print(f"Mean Abs Err:   {df.abs_err.mean():.2f}")
    print(f"R² (pred vs actual): {r2_score(df.actual, df.pred):.3f}")

    df.to_csv(args.out_csv,index=False)
    print(f"\nSaved → {args.out_csv}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Evaluate YOLO symbiont detection vs CellCounter XML truth.")
    ap.add_argument("--in_dir",required=True,help="Path to folder containing training JPG + XML files")
    ap.add_argument("--roi_json",required=True,help="Path to ROI cache JSON")
    ap.add_argument("--weights",required=True,help="Path to trained YOLO weights (.pt)")
    ap.add_argument("--imgsz",type=int,default=1280,help="Image size for YOLO model")
    ap.add_argument("--conf",type=float,default=0.35,help="Confidence threshold")
    ap.add_argument("--iou",type=float,default=0.5,help="IOU threshold")
    ap.add_argument("--match_px",type=int,default=10,help="Distance tolerance (px) for matching detections to true cells")
    ap.add_argument("--out_csv",default="evaluation_training.csv",help="Path to save CSV results")
    args=ap.parse_args()
    main(args)
