# Dataset Download Guide

## Quick Start

1. **Get your Roboflow API key** from [roboflow.com/settings/api](https://roboflow.com/settings/api)

2. **Install dependencies** (if not already installed):
   ```bash
   pip install roboflow
   ```

3. **Download all 4 datasets**:
   ```bash
   python scripts/download_datasets.py --all --api-key YOUR_API_KEY
   ```

## Individual Crop Downloads

```bash
# Tomato (513 images, 1 class)
python scripts/download_datasets.py --crop tomato --api-key YOUR_API_KEY

# Strawberry (4,000 images, 3 classes)
python scripts/download_datasets.py --crop strawberry --api-key YOUR_API_KEY

# Apple (10,000 images, 1 class)
python scripts/download_datasets.py --crop apple --api-key YOUR_API_KEY

# Kiwi (302 images, 1 class) — HELD-OUT for final evaluation
python scripts/download_datasets.py --crop kiwi --api-key YOUR_API_KEY
```

## Using Environment Variable

For convenience, store your API key:
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
python scripts/download_datasets.py --all
```

**Or use the setup script** (interactive):
```bash
bash scripts/setup_roboflow_auth.sh
python scripts/download_datasets.py --all
```

## Troubleshooting

### API Key Not Found / Invalid Key
**Error**: `This API key does not exist (or has been revoked)`

**Solution**:
1. Log in to [roboflow.com](https://roboflow.com)
2. Go to **Settings → API Keys**
3. Under "Private API Key", copy the **full key** (ensure it's not truncated)
4. Generate a new key if needed by clicking the refresh icon
5. Try again:
   ```bash
   python scripts/download_datasets.py --all --api-key YOUR_NEW_KEY
   ```

### Roboflow CLI Not Installed
**Error**: `roboflow CLI not found`

**Solution**:
```bash
pip install roboflow
```

### Authentication Fails
1. Double-check your API key is fully copied (not truncated in the screenshot)
2. Verify your account has Public Plan (minimum requirement)
3. Try regenerating a new key in the Roboflow dashboard

## Dataset Output

Datasets are downloaded to `data/{crop_name}/` in YOLO format:
```
data/
├── tomato/
│   ├── images/{train,val,test}/*.jpg
│   └── labels/{train,val,test}/*.txt (YOLO format)
├── strawberry/
├── apple/
└── kiwi/ (HELD-OUT — never train until Step 5)
```

## Validation

After downloading, validate label format:
```bash
python -m greenpoll.data.validate_yolo_labels data/tomato 1
```

## Notes

- **Roboflow API Key**: Required to download datasets. Free account available at roboflow.com
- **Kiwi Freeze Policy**: Kiwi is held-out as the target crop for cross-crop transfer evaluation. Do NOT use it for training until Step 5 final evaluation.
- **Storage**: Downloads total ~15GB. Ensure sufficient disk space.
- **Network**: Large downloads (especially Apple 10k images). Expect 10-30 minutes per crop depending on connection speed.
