# sym_counting_ultralytics

#Script by Maya Powell
#Created: November 4th, 2025
#Last edited: September 10th, 2026
#Created using the help of generative AI - Claude Sonnet 4.5

# General information
1. These scripts are created for processing hemocytometer images of coral symbiont cells to generate accurate cell counts
2. These can be run either on a computing cluster (recommended) your personal computer (not recommended)
3. The pipeline below is a general walkthrough and will need to be edited for your personal dataset and computing setup
4. These were generated for use in linux/unix as python scripts

# What you need to replace to utilize this pipeline
1. Folder of images you are interested in processing (here I have included a very small sample of this (all_images_example)
2. Training dataset images - a subset of 10% of your full image dataset, make sure to include representative samples if you have any differences in image quality/exposure/coral species/etc. (training_dataset)
3. Training dataset cell counts from Fiji (Image J). Create this by opening your training datasets one by one in Image J. For each image go to Plugins > Analyze > Cell counter. A new counter window will open. To begin counting, click ‘Initialize’ then ‘Type 1’ in the cell counter window. Once you are done, click "Save Markers" and save them to your training dataset folder.

# Example process
1. Create and activate a conda environment
```
conda create symcount
conda activate symcount
```

2. Navigate to the folder where you are doing work
```
cd sym_counting_ultralytics
```

3. Install or activate packages and make sure all versions are correct

```
export PYTHONNOUSERSITE=1   # avoid user-site packages leaking in
```

install pandas
```
pip install pandas
```

for the rest of the installation
sanity: confirm we’re in the right interpreter
```
python -c "import sys; print(sys.executable)"
```
this should show you the correct filepath

pin NumPy to 1.x FIRST (prevents later upgrades)
```
python -m pip install "numpy==1.26.4"
```

install a CPU torch build that works with NumPy 1.x
```
python -m pip install --no-cache-dir \
  "torch==2.2.2" "torchvision==0.17.2" \
  --index-url https://download.pytorch.org/whl/cpu
  ```

install Ultralytics WITHOUT pulling deps (so it can’t bump numpy)
```
python -m pip install --no-deps "ultralytics==8.3.225"
```

install exactly ONE OpenCV (headless is safest for servers/conda)
```
python -m pip install "opencv-python-headless==4.10.0.84" matplotlib PyYAML tqdm psutil
```

verify versions (these must print successfully)
```
python - <<'PY'
import numpy, torch, ultralytics, cv2
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("ultralytics", ultralytics.__version__)
print("opencv", cv2.__version__)
PY
```

these are the versions you should have:
1. numpy 1.26.4
2. torch 2.2.2
3. ultralytics 8.3.225
4. opencv 4.10.0

4. Create roi (region of interest) box for training dataset- here that is the hemocytometer 5x5 grid
This prompts you to click the top left and bottom right to generate the roi square and set your scale

```
python roi_prepare_from_xml_v2.py \
  --in_dir  ~/sym_counting_ultralytics/training_dataset \
  --out_dir ~/sym_counting_ultralytics/yolo_data_b16 \
  --roi_json ~/sym_counting_ultralytics/roi_cache.json \
  --box_um 10 \
  --val_frac 0.15 \
  --filter_to_roi true
```  
  
and then outputs the overlays with your cells that you selected as true in this folder:

yolo_data_b16/overlays/<image>_truth.png

examine these for a quick visual qc to make sure your cell counter info was correctly imported and to check the size of the Xs.
If the boxes for your specific symbiont type look a bit small/large, re-run with --box_um 8 or --box_um 12, etc.

3. Train a small detector (ultralytics)
This will take a long time (single thread took ~11 hours for 120 images)

```
export OMP_NUM_THREADS=1 #change number of threads to increase speed
```

```
RUNNAME=train_y8s_e100_b16_1280_2Sept2026 #define run name with info
```

```
yolo detect train \
  model=yolov8s.pt \
  data=~/sym_counting_ultralytics/yolo_data_b16/sym_dataset.yaml \
  imgsz=1280 \
  epochs=100 \
  batch=4 \
  mosaic=0 degrees=5 translate=0.03 scale=0.12 shear=0.0 \
  hsv_h=0.0 hsv_s=0.12 hsv_v=0.12 \
  fliplr=0.0 \
  workers=0 \
  patience=50 \
  name=$RUNNAME
```
  
If recall is low, increase epochs (e.g to 120) and/or increase imgsz

4. Run the detector on your training set to evaluate itʻs efficacy 

from before: RUNNAME=train_y8s_e100_b16_1280_2Sept2026

```
python infer_and_count.py \
  --in_dir  ~/sym_counting_ultralytics/training_dataset \
  --out_dir ~/sym_counting_ultralytics/detections_y8s_e100_b16_1280_2Sept2026 \
  --roi_json ~/sym_counting_ultralytics/roi_cache.json \
  --weights  ~/sym_counting_ultralytics/runs/detect/$RUNNAME/weights/best.pt \
  --imgsz 1280 --conf 0.35 --iou 0.50
```
  
if it is underestimating, lower conf (e.g. to 0.3), or overestimating raise (e.g. to 0.4). 

I ran multiple times to test and 0.35 was best for my dataset

5. Evaluate training run 
This uses a spatial evaluation script to get both false positive and false negative counts

```
python evaluate_spatial.py \
  --in_dir  ~/sym_counting_ultralytics/training_dataset \
  --roi_json ~/sym_counting_ultralytics/roi_cache.json \
  --weights  ~/sym_counting_ultralytics/runs/detect/train_y8s_e100_b16_1280_2Sept2026/weights/best.pt \
  --imgsz 1280 --conf 0.35 --iou 0.5 --tta \
  --match_um 6 \
  --out_csv ~/sym_counting_ultralytics/eval_spatial_training_2sept2026.csv \
  --out_dir ~/sym_counting_ultralytics/eval_spatial_debug_2sept2026 \
  --save_overlays
```
  
and in case you have to reclick the ROI for any individual images that were mistakes or weird:

add in the --reclick argument e.g.:
```
  --reclick P8174886 \
```
  
6. Run new samples
I would test this on a small image set of ~10 images first to make sure you are satisfied with the performance before completing your entire dataset

```
python infer_and_count.py \
  --in_dir  ~/sym_counting_ultralytics/all_images_example \
  --out_dir ~/sym_counting_ultralytics/all_image_detections_example \
  --roi_json ~/sym_counting_ultralytics/roi_cache.json \
  --weights  ~/sym_counting_ultralytics/runs/detect/train_y8s_e100_b16_1280_2Sept2026/weights/best.pt \
  --imgsz 1280 \
  --conf 0.35 \
  --iou 0.5 \
  --tta 
```


while I was running this on my personal computer it stopped mid-run because I ran out of space
to clear this I removed my cache here:

```
rm -f "$HOME/Library/Application Support/Ultralytics/persistent_cache.json"


7. Extract counts
Within your --out_dir (here: all_image_detections_example) there is an excel spreadsheet labeled "counts"
You can export this counts spreadsheet and align it with your metadata to analyze symbiont density and average across replicates etc.
In this folder, you can also examine all of your image detections to assess how well the model worked on your counts and adjust your model as needed

# Additional notes on how each script works:

roi_prepare_from_xml_v2.py (dataset & ROI prep)

Purpose: turn your manually annotated images (Fiji CellCounter XML + raw JPG) into a clean YOLO dataset and an ROI cache.

What it does: 
1. Reads CellCounter XML (CellCounter_<stem>.xml), pulls Type-1 markers (your blue-dot truths), and pairs them to the matching raw image.
2. Builds/updates an ROI cache (roi_cache.json).
3. If an ROI is missing or invalid, it can prompt you to click top-left and bottom-right of the 5×5 grid (1 mm wide → 25 squares).
4. Transforms ground-truth points to YOLO labels inside the ROI:
5. Converts dot annotations to small boxes centered on each dot (YOLO needs boxes, not points).
6. Writes one labels/<stem>.txt per image (class, x_center, y_center, width, height in normalized ROI coordinates).
7. Splits images and labels into yolo_data/train/ and yolo_data/val/ (stratified or simple split).
8. (Optional) QA overlays so you can visually verify ROI and label placement.

Key inputs: raw JPGs, CellCounter_*.xml.
Outputs: yolo_data/{images,labels}/{train,val}, roi_cache.json, optional overlays.

Knobs you can tweak: 
1. Train/val split fraction.
2. The box size used to convert dots → YOLO boxes (slightly larger boxes can help the detector learn tiny objects).
3. Whether to re-click ROIs or reuse cached ones.

yolo detect train (model training)

Purpose: train a detector to find symbiont cells inside the ROI crops.

What it does:
1. Loads a YOLOv8 model (e.g., yolov8s.pt or yolov8n.pt) and your sym_dataset.yaml pointing to yolo_data/.
2. Trains for N epochs with your chosen image size and augmentations.
3. Typical command:
```
yolo detect train \
  model=yolov8s.pt \
  data=.../yolo_data/sym_dataset.yaml \
  imgsz=1280 \
  epochs=180 \
  batch=16 \
  mosaic=0 degrees=5 translate=0.05 scale=0.10 hsv_h=0.0 hsv_s=0.15 hsv_v=0.15 fliplr=0.0 \
  workers=0
```

Why these settings
1. imgsz=1280: more pixels helps with small, round cells.
2. mosaic=0 and gentle geometric/color augs: preserve the fine texture and roundness; avoid heavy mosaics that distort scale.
3. Small model (y8n/y8s): good balance for CPU/mac training.

Outputs: runs/detect/train/* with best.pt, plots (results.png, labels.jpg), tensorboard logs, and validation metrics.

What to monitor
1. Val plots and labels.jpg (sanity check).
2. If the model undercounts, consider:
- Lower conf or increase box size during label generation,
- Train longer or move from y8n → y8s → y8m if you have compute,
- Add more diverse training images.

infer_and_count.py (production inference & counting)

Purpose: run the trained model on new images, restricted to the 5×5 ROI, and save both overlays and a counts.csv.

What it does:
1. Loads ROI cache (roi_cache.json). If an image is missing or has a bad ROI, it prompts you to click TL/BR of the 5×5 grid (with R to reuse last box).
2. Crops the image to the ROI, runs YOLO, and collects detections.
3. Writes an overlay (raw image + red ROI box + green boxes) and appends a row to counts.csv with image_id and count.
4. Supports --tta (test-time augmentation) to boost recall for faint/partial cells.
5. Typical command:
```
python infer_and_count.py \
  --in_dir  ~/.../new_images \
  --out_dir ~/.../detections_final \
  --roi_json ~/.../roi_cache.json \
  --weights  ~/.../runs/detect/train_y8s_e180_b16_1280/weights/best.pt \
  --imgsz 1280 \
  --conf 0.35 \
  --iou 0.5 \
  --tta
```

Key parameters
--conf: confidence threshold; your runs suggested 0.35–0.40 is the sweet spot (lowest MAE).
--tta: on for higher recall (slower).

ROI cache is written atomically to avoid corruption; you can re-pick any image’s ROI as needed.

Outputs
1. Overlays: <image>_det.png (raw image with ROI and detections).
2. counts.csv: tidy table for downstream merging.

How the pieces fit together
1. roi_prepare_from_xml_v2.py: builds clean training data + ROI cache from your Fiji annotations.
2. yolo detect train: learns to detect symbionts within ROI crops using your curated labels.
3. evaluate_spatial.py: spatial TP/FP/FN by matching detections to ground-truth points (with ROI re-picker and µm→px matching tolerance).
3. infer_and_count.py: applies the trained detector to new ROIs, saves overlays, and counts.

For spatial training eval (e.g. eval_spatial_training_2Sept2026.csv)
| **Column**  | **Meaning**                                                                                                    | **Units / Notes** |   |       |
| ----------- | -------------------------------------------------------------------------------------------------------------- | ----------------- | - | ----- |
| `image`     | The base filename (stem) of the image analyzed (e.g. `P8164337`)                                               | –                 |   |       |
| `actual`    | The total number of **true symbiont cells** in the ROI from your Fiji CellCounter XML                          | count             |   |       |
| `pred`      | The number of **cells detected** by your trained YOLO model in the ROI                                         | count             |   |       |
| `tp`        | **True Positives** — model detections that correctly matched a true cell within the tolerance radius           | count             |   |       |
| `fp`        | **False Positives** — model detections that did *not* correspond to any true cell                              | count             |   |       |
| `fn`        | **False Negatives** — true cells that the model *missed*                                                       | count             |   |       |
| `precision` | Fraction of detections that are correct: `TP / (TP + FP)`                                                      | 0–1               |   |       |
| `recall`    | Fraction of true cells correctly detected: `TP / (TP + FN)`                                                    | 0–1               |   |       |
| `f1`        | Harmonic mean of precision and recall: `2PR / (P + R)`                                                         | 0–1               |   |       |
| `abs_err`   | Absolute error in count: `                                                                                     | pred - actual     | ` | count |
| `diff`      | Signed difference between predicted and actual counts (`pred - actual`)                                        | count             |   |       |
| `tol_px`    | The matching tolerance used for spatial pairing, in **pixels** (converted from your `--match_um` if specified) | pixels            |   |       |

