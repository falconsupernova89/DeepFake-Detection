# DeepFake Detector — Streamlit App

## Setup

1. Place these files in one folder:
   - app.py
   - requirements.txt
   - deepfake_model.pth  ← your trained weights

2. Install dependencies:
   pip install -r requirements.txt

3. Run:
   streamlit run app.py

## Usage
- Select Image or Video mode
- Upload a file
- Adjust threshold (default 0.35)
- Click ANALYSE

## Notes
- Runs fully on CPU — no GPU needed
- Video: samples up to 50 frames, every 3rd frame
- Ensemble = 60% EfficientNet-B4 DNN + 40% 6 forensic signals
