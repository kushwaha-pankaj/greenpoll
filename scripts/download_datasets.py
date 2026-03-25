#!/usr/bin/env python3
"""Download crop flower datasets from Roboflow Universe.

Usage:
    python scripts/download_datasets.py --crop tomato
    python scripts/download_datasets.py --all
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Roboflow dataset config: {crop: (workspace/project/version, format)}
DATASETS = {
    'tomato': ('stefs/tomato-flower-detection-zmeju/1', 'yolov8'),
    'strawberry': ('noc/strawberry-flower/1', 'yolov8'),
    'apple': ('flowers-rhffp/apple-flower-sahi/1', 'yolov8'),
    'kiwi': ('timm-brx5x/kiwi-flower-recognition/1', 'yolov8'),
}

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def download_crop(crop_name: str, api_key: str = None) -> bool:
    """Download a single crop dataset from Roboflow."""
    if crop_name not in DATASETS:
        print(f'Unknown crop: {crop_name}. Choose from: {list(DATASETS.keys())}')
        return False
    
    dataset_url, fmt = DATASETS[crop_name]
    out_path = DATA_DIR / crop_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Downloading {crop_name} from {dataset_url}...')
    cmd = [
        'roboflow', 'download',
        dataset_url,
        '-f', fmt,
        '-l', str(out_path),
    ]
    if api_key:
        cmd.extend(['--api-key', api_key])
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f'✓ {crop_name} dataset downloaded to {out_path}')
        return True
    except subprocess.CalledProcessError as e:
        print(f'✗ Failed to download {crop_name}: {e}')
        return False
    except FileNotFoundError:
        print('roboflow CLI not found. Install with: pip install roboflow')
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download crop datasets from Roboflow.')
    parser.add_argument('--crop', type=str, help='Specific crop to download (tomato/strawberry/apple/kiwi)')
    parser.add_argument('--all', action='store_true', help='Download all crops')
    parser.add_argument('--api-key', type=str, default=None, help='Roboflow API key (default: from env)')
    args = parser.parse_args()
    
    if not args.crop and not args.all:
        parser.print_help()
        sys.exit(1)
    
    crops_to_download = list(DATASETS.keys()) if args.all else [args.crop]
    success_count = 0
    for crop in crops_to_download:
        if download_crop(crop, args.api_key):
            success_count += 1
    
    print(f'\n{success_count}/{len(crops_to_download)} datasets downloaded successfully.')
    sys.exit(0 if success_count == len(crops_to_download) else 1)
