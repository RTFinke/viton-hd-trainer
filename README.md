
# VITON-HD Training for CatVTON

## Setup

```bash
conda create -n vitonhd python=3.8 -y
conda activate vitonhd
pip install -r requirements.txt
```

## Dataset structure

```
data/
├── train_img/
├── train_cloth/
├── train_cloth_mask/
├── train_parse/
├── test_img/
├── test_cloth/
├── test_cloth_mask/
├── test_parse/
```

## Run training

```bash
python train_viton_hd.py \
  --dataroot ./data \
  --name vitonhd_run1 \
  --batch_size 4 \
  --epochs 100
```

## Output

Models are saved to:
```
checkpoints/vitonhd_run1/
```
