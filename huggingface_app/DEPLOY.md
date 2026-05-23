# Deploying to Hugging Face Spaces

## Step-by-Step Instructions

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up for a free account

### 2. Create New Space
- Go to https://huggingface.co/new-space
- **Space name:** `marathon-predictor` (or your choice)
- **SDK:** Select `Gradio`
- **Hardware:** `CPU Basic` (free tier)
- **Visibility:** Public or Private
- Click "Create Space"

### 3. Upload Files

Option A: **Via Web Interface**
1. Click "Files" tab in your Space
2. Click "Add file" → "Upload files"
3. Upload these files from `huggingface_app/`:
   - `app.py`
   - `model.pkl`
   - `requirements.txt`
   - `README.md`

Option B: **Via Git**
```bash
# Clone your HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/marathon-predictor
cd marathon-predictor

# Copy files
cp /path/to/strava_guru/huggingface_app/* .

# Push
git add .
git commit -m "Initial deployment"
git push
```

### 4. Wait for Build
- Hugging Face will automatically build and deploy
- Takes 2-5 minutes
- Check "Logs" tab for progress

### 5. Test Your App
- Once deployed, your app will be at:
  `https://huggingface.co/spaces/YOUR_USERNAME/marathon-predictor`

## Updating the App

To update after changes:
```bash
cd marathon-predictor
git pull
# Make changes
git add .
git commit -m "Update description"
git push
```

## Troubleshooting

### "No module named X"
- Check `requirements.txt` includes all dependencies
- Gradio, pandas, numpy, scikit-learn should be listed

### Model loading fails
- Ensure `model.pkl` was uploaded
- Check file isn't corrupted

### App crashes on upload
- Check Strava zip format
- Verify activities.csv exists in zip

## Files Included

```
huggingface_app/
├── app.py           # Main Gradio application
├── model.pkl        # Trained Random Forest model (48 features)
├── requirements.txt # Python dependencies
├── README.md        # HuggingFace Space metadata
└── DEPLOY.md        # This file
```
