🐖 HogIntel — Two-Stage Weight → Price System

HogIntel is a two-step pipeline for hog weight prediction and market price estimation, designed to give users full control and higher reliability.

🔹 How It Works

Stage 1 — Weight Screen
📸 Capture image → 🐖 Detect hog → ✂️ Crop ROI → ⚖️ Predict weight
➡️ User can Accept / Retake / Adjust

Stage 2 — Price Screen
After confirmation → 📊 Predict market price (via XGBoost) → 💰 Compute value

Separating weight and price prevents errors from propagating and improves accuracy.

🔹 Features

🧩 YOLOv8 — hog detection & segmentation

🧠 CNN regression — weight estimation (target MAE < 3kg)

✅ User confirmation flow — Accept / Retake / Adjust

💵 Price module — powered by XGBoost (with Prophet fallback option)

📡 Telemetry storage — for retraining & improvement

🔹 Project Structure
hogintel/
├─ app/
│  ├─ main.py              # FastAPI entrypoint
│  ├─ config.py            # Env vars & calibration defaults
│  ├─ schemas.py           # Pydantic request/response
│  ├─ logger.py            # Logging utility
│  ├─ utils/               # Preprocessing & helpers
│  │   ├─ image_utils.py   # ROI extraction, transforms
│  │   ├─ camera_utils.py  # Distance calibration
│  │   └─ metrics.py       # MAE, RMSE, R²
│  ├─ models/              # ML wrappers
│  │   ├─ yolo_detector.py # YOLOv8 hog detection
│  │   ├─ cnn_regressor.py # Predict hog weight
│  │   └─ price_model.py   # Predict hog price (XGBoost or other)
│  └─ routers/             # API endpoints
│      ├─ scan.py          # /scan → weight inference
│      ├─ confirm.py       # /confirm → adjust/store weight
│      └─ price.py         # /price → price prediction
│
├─ training/               # Training scripts
│  ├─ train_yolo.py
│  ├─ train_cnn.py
│  └─ train_price.py
│
├─ models/                 # Pretrained artifacts (.pt/.onnx/.json)
│  ├─ hog_yolo.pt
│  ├─ cnn_regressor.pt
│  └─ price_model.json
│
├─ data/                   # Datasets
│  ├─ raw/                 # Raw images
│  ├─ annotations/         # Label Studio exports
│  └─ calibration.json     # Camera calibration
│
├─ scripts/                # CLI utilities
│  ├─ inference_cli.py     # Run local inference
│  └─ export_dataset.py    # Convert labels → training format
│
├─ docker/                 # Deployment setup
│  ├─ Dockerfile
│  └─ docker-compose.yml
│
├─ requirements.txt
└─ README.md
🔹 Workflow

📸 Capture Image — via camera app + calibration marker
⚖️ Weight Prediction — YOLOv8 ROI → CNN regression
✅ User Confirmation — Accept / Retake / Adjust
💵 Price Estimation — triggered only after confirmation

Lookup (historic DB)

Predictive model (XGBoost, Prophet fallback)
📊 Result Display — price per kg + total value

🔹 API Endpoints

POST /api/v1/scan → Run weight inference

POST /api/v1/confirm_weight → Store confirmed weight

POST /api/v1/price → Estimate price (after confirmation)

🔹 Frontend (Flutter)

Screen A — Weight

Capture → call scan API → show weight + confidence

Options: Accept | Retake | Adjust

On Accept → call confirm_weight → go to Price screen

Screen B — Price

Uses confirmed weight

Calls price API → shows price per kg + total value

🔹 Evaluation Targets

🎯 Detection (YOLOv8): mAP > 0.90
🎯 Weight model (CNN): MAE < 3kg, R² > 0.92
🎯 Price model (XGBoost): ≤10% error vs baseline
🎯 System: ~97% combined accuracy with proper dataset + calibration