# Project Notes

## CrossPoll — Key Reference Notes

### Research Question
Can a detector jointly pretrained on flowers from multiple crops adapt to a new crop faster (fewer labels) than standard baselines?

### 4 Baselines
1. **Scratch** — Train YOLOv8n from random init on kiwi only
2. **COCO Transfer** — Fine-tune `yolov8n.pt` (COCO pretrained) on kiwi
3. **Single-Crop Transfer** — Pretrain on 1 source crop → fine-tune on kiwi
4. **CrossPoll (ours)** — Joint pretrain on 3 crops → fine-tune on kiwi

### Held-out Crop: Kiwi
- 302 images total (train=181, valid=60, test=61)
- 1 class: `kiwi-flower`
- Label budgets: [25, 50, 100, 200, 500] — cap at 181 for budgets > available

### Training Parameters
- Model: YOLOv8n | Epochs: 300 | Patience: 20 | Image size: 640
- Optimizer: AdamW | AMP: enabled | Batch: auto
- LR (joint): 0.001 | LR (finetune): 0.0001

### Metrics
- Primary: mAP@50
- Secondary: Precision, Recall, F1
- Efficiency: training time, model size, inference latency
