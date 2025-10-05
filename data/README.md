Data Collection Plan — HogIntel (goal: push toward 97% accuracy)
1) Overall targets (end goal)

Total images: 8,000 – 12,000 labeled images (minimum viable: 2,000; good: 5,000; ideal: 10k+).

Unique animals: at least 2,000 unique hogs (multiple shots per hog across poses).

Paired weight records: 100% of images must link to a ground-truth weight measured on a scale (kg).

Price history rows: full monthly price data for each target region covering last 3–5 years.

2) Why these numbers

Detection (YOLOv8) generalizes well with several thousand labeled instances.

Weight regression needs many variations in pose, distance, lighting, and breeds to avoid overfitting — hence 8k–12k images.

3) Capture protocol (what each photo should look like)

Make a short checklist for each capture; collectors should follow this exact sequence:

Weigh the hog on a calibrated scale — record weight to 1 decimal (e.g., 78.5 kg).

Place a calibration marker (a board/marker of known width, e.g., 30 cm) visible in the scene near the hog (not obscuring). If possible, measure and record the marker pixel width after capture.

Take 4 photos per hog:

Frontal (head-on) at ~2–3 m.

Side (left) full body at ~2–3 m.

Side (right) full body at ~2–3 m.

Top/angled / closer crop (if safe) or distance variant (to capture scale variance).

Vary distances for different hogs (1.5m, 2m, 3m) so model learns scale — but always include calibration marker.

Lighting: take samples in bright daylight, cloudy, and low-light conditions. Aim for 60% daylight, 30% cloudy/indoor, 10% low-light.

Occlusion/pose: include some images where part of the hog is occluded by pen bars or other hogs (not >30% occlusion).

Background diversity: pens with different colors, outdoor dirt, concrete, grass.

Record metadata (see next section) immediately after capture.

Safety / animal welfare

Do not stress animals; only standard farm handling. If a hog is stressed, skip.

4) Metadata (mandatory columns per capture)

Store as a CSV or JSONL entry for each image:

image_id (UUID)

file_path

hog_id (unique per animal)

capture_timestamp (ISO8601)

weight_kg (ground-truth scale)

user_id (collector)

region (e.g., MetroManila)

camera_model (phone/tablet)

camera_focal_length (if available)

marker_real_width_cm (e.g., 30)

marker_pixel_width (measured after capture or computed later)

distance_estimate_m (if using a laser / phone rangefinder)

lighting (categories: daylight/cloudy/lowlight)

pose (frontal/left/right/angled/close)

notes (occlusion, multiple hogs, stress)

5) Annotation instructions (Label Studio config + rules)

Annotations per image:

Bounding box around each visible hog (label Hog).

Segmentation polygon for hog body mask (label HogMask).

Keypoints (optional but recommended): Snout, Withers (shoulder), TailBase, Belly.

Labeling rules:

If multiple hogs in image, annotate each as a separate object and indicate primary_hog (the one with weight label).

For masks, follow body outline including legs, exclude shadows.

For keypoints, place within a few pixels of landmark — if not visible, leave blank.

Minimum annotation quality: IoU > 0.85 for polygon vs. reference for QA; inter-annotator IoU target > 0.90.

Use your Label Studio XML as in the guide (bbox, polygon, keypoints). Add an attribute hog_id to connect to recorded weight.

6) Calibration data

For each camera type used (e.g., phone model), collect a camera calibration set: photograph a calibration board (known dimensions) at multiple distances (1m, 2m, 3m). Save marker_pixel_width for each distance. This helps compute scale factor during inference or training.

7) Dataset splits & augmentation

Train / Val / Test split by hog_id (important — ensure the same animal doesn’t appear across splits):

Train: 80% of hogs

Val: 10%

Test: 10%

Augmentation for training only:

Random brightness/contrast, horizontal flip, small rotations (±10°), slight scaling (±15%), blur, occlusion augmentation (simulate pens).

Balanced weight distribution: ensure all weight ranges represented (bins e.g., <30kg, 30–60, 60–90, 90–120, >120). If underrepresented, oversample or collect targeted images.

8) Quality assurance (QA)

Labeling QA workflow:

Each image annotated by 1 labeler.

Random 10% double-labeled by a second annotator for inter-annotator agreement (compute IoU and MAE on manual keypoint placement).

If disagreement > threshold (IoU < 0.85), send back to reviewer.

Weight data QA:

Calibrate scales weekly. Keep weight logs with scale ID and calibration certificate.

Automated checks:

Missing metadata rows flagged.

Images where marker_pixel_width missing flagged.

Outlier weight values compared to predicted weight (after initial model run) flagged for re-inspection.

9) Collection logistics & personnel prompts

Team: 2–4 field collectors (phones + scale), 2 annotators, 1 QA reviewer.

Daily target: 50–150 images/day per collector (depends on access).

Estimated time to reach 10k images: ~8–12 weeks with 2 collectors active and concurrent annotation.

Field script / prompt (for collectors)

Use this short script on a tablet or printed card:

“Weigh hog on scale → record weight to 0.1 kg.”

“Place marker (30 cm) near hog, visible.”

“Take 4 shots: front, left, right, close/angled.”

“Capture metadata: hog_id, region, lighting, notes.”

“Upload images and metadata to server.”

“If hog stressed, wait until calm.”

10) Privacy & consent

If collecting on private farms, obtain owner consent. Store a short consent form: farm name, owner contact, date, signature/photo of consent.

11) Storage & backup

Use object storage (S3 or MinIO) for images. Keep metadata in a transactional DB (Postgres). Daily backups and checksum verification.

12) Labeling tool & settings

Label Studio project:

Use your XML config with extra metadata fields shown.

Use hotkeys and annotation templates for speed.

Export in COCO format for YOLO conversion; keep mask PNGs for segmentation.

13) Telemetry & continuous improvements

Whenever user confirms weight in production, store predicted_weight, confirmed_weight, image_id — feed this back monthly into a retraining pipeline.

Retrain schedule: every 4–8 weeks with new confirmed samples.

14) Example dataset CSV schema (header)
image_id,file_path,hog_id,capture_timestamp,weight_kg,user_id,region,camera_model,marker_real_width_cm,marker_pixel_width,distance_estimate_m,lighting,pose,notes

15) Minimum viable rollout (fastest path)

If you need a faster start with fewer resources:

Collect 2,000 images with full metadata and strict calibration markers.

Annotate and train YOLO + CNN. Expect ~85–90% initially. Use telemetry to iterate.

Quick prioritized checklist (what to do first)

Prepare scale and calibration marker kits.

Train collectors with the field script. Run a 2-day pilot (collect 200–300 images).

Set up Label Studio project and annotate pilot batch; run initial YOLO trial.

Adjust capture protocol based on pilot failure modes (lighting, occlusion).

Ramp to full collection.