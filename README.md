VITON-HD Training for CatVTON
This repository provides an implementation of VITON-HD for virtual try-on, adapted for the CatVTON framework. It is based on the paper: Diffusion VTON: High-Fidelity Virtual Try-On Network via Mask-Aware Diffusion Model.
Overview
This project allows you to train a high-fidelity virtual try-on model using the VITON-HD dataset. The model warps clothing images to match a person's pose, generates masks, and produces realistic try-on images using a GAN-based architecture.
Requirements

Hardware:
NVIDIA GPU with 16GB+ VRAM (8GB may work with reduced batch_size or img_size=256).
Multi-core CPU for data loading.
32GB+ RAM recommended.


Software:
Python 3.8+
CUDA 11.0+ (if using GPU)
Conda (for environment management)



Installation

Create and activate a Conda environment:
conda create -n vitonhd python=3.8 -y
conda activate vitonhd


Install dependencies:
pip install -r requirements.txt


Verify installation:
python -c "import torch; print(torch.cuda.is_available())"

This should print True if CUDA is properly set up.


Dataset Preparation
The VITON-HD dataset is required for training and testing. You can download it from the official VITON-HD repository or other sources providing the dataset. The dataset should have the following structure:
data/
├── train_img/        # Person images (.jpg, .png, .jpeg)
├── train_cloth/      # Cloth images (.jpg, .png, .jpeg)
├── train_cloth_mask/ # Cloth masks (.png)
├── train_parse/      # Person parsing/segmentation (.png)
├── test_img/         # Person images for testing
├── test_cloth/       # Cloth images for testing
├── test_cloth_mask/  # Cloth masks for testing
├── test_parse/       # Person parsing for testing

Steps to Set Up the Dataset

Download VITON-HD:

Obtain the dataset from the official source or a trusted provider.
The dataset typically comes in a compressed format (e.g., .zip or .tar.gz).


Extract the Dataset:

Unzip the dataset to a directory, e.g., ./data/.
Ensure the folder structure matches the one above. For example:./data/train_img/person_001.jpg
./data/train_cloth/cloth_001.jpg
./data/train_cloth_mask/cloth_001.png
./data/train_parse/person_001.png




Place the Dataset in the Project:

Move the extracted dataset to the ./data/ folder in your project directory.
If the dataset has a different folder structure (e.g., train/image/ instead of train_img/), rename the folders to match the expected structure.


Validate the Dataset:

Run the validation script to check if all required files are present and correctly formatted:python scripts/validate_data.py --dataroot ./data


If errors are reported (e.g., missing files), fix the dataset structure or download missing components.



Notes on VITON-HD Dataset

Image Formats: The code supports .jpg, .png, and .jpeg for images and expects .png for masks and parsing.
File Naming: Ensure file names are consistent across directories (e.g., person_001.jpg in train_img/ should have a corresponding person_001.png in train_parse/).
Size: Images are resized to 512x512 (or 256x256 if specified) during training, so ensure the dataset images are of sufficient resolution.

Project Structure
After setting up the dataset, your project directory should look like this:
viton-hd/
├── data/
│   ├── train_img/
│   ├── train_cloth/
│   ├── train_cloth_mask/
│   ├── train_parse/
│   ├── test_img/
│   ├── test_cloth/
│   ├── test_cloth_mask/
│   ├── test_parse/
├── models/
│   ├── __init__.py
│   ├── appearance_flow.py
│   ├── mask_generator.py
│   ├── networks.py
├── scripts/
│   ├── validate_data.py
├── checkpoints/       # Model checkpoints will be saved here
├── results/          # Inference results will be saved here
├── train_viton_hd.py
├── inference.py
├── evaluate.py
├── dataset.py
├── requirements.txt
├── README.md

Training
To train the model:
python train_viton_hd.py \
  --dataroot ./data \
  --name vitonhd_run1 \
  --batch_size 2 \
  --img_size 256 \
  --epochs 50 \
  --log_wandb

Key Arguments

--dataroot: Path to the dataset (default: ./data).
--name: Experiment name (checkpoints saved to checkpoints/<name>).
--batch_size: Batch size (use 2 or 1 for GPUs with <16GB VRAM).
--img_size: Image size (256 for faster training, 512 for higher quality).
--epochs: Number of training epochs.
--log_wandb: Enable Weights & Biases logging (requires wandb login).

Tips for Training

Low VRAM: If you encounter out-of-memory errors, reduce batch_size to 1 or set img_size=256.
WandB: Run wandb login before enabling --log_wandb, or omit the flag to disable制作人: xAI disable it if not needed.
Checkpoints: Models are saved to checkpoints/<name>/ every 10 epochs and at the end of training.

Inference
To generate try-on images using a trained model:
python inference.py \
  --checkpoint checkpoints/vitonhd_run1_epoch_49.pth \
  --dataroot ./data \
  --output_dir ./results \
  --img_size 256

Results are saved to ./results/ as PNG files named <person_name>_<cloth_name>.png.
Evaluation
To evaluate the model on the test set (SSIM and PSNR metrics):
python evaluate.py \
  --checkpoint checkpoints/vitonhd_run1_epoch_49.pth \
  --dataroot ./data \
  --batch_size 2 \
  --img_size 256

Troubleshooting

Out of memory errors: Reduce batch_size or img_size. For 8GB GPUs, try batch_size=1 and img_size=256.
Missing dataset files: Run python scripts/validate_data.py --dataroot ./data to identify and fix issues.
WandB errors: Ensure wandb login is executed or disable --log_wandb.
CUDA errors: Verify CUDA installation and GPU compatibility. Use CPU as a fallback (slower) by setting CUDA_VISIBLE_DEVICES="".
No valid data samples: Check dataset structure and file formats. Ensure all required files (images, masks, parsing) are present.

Additional Resources

VITON-HD Paper: Read the original paper for technical details.
Official Repository: Visit VITON-HD GitHub for dataset and reference code.
WandB Setup: See WandB documentation for logging setup.

If you encounter issues or need further assistance, feel free to open an issue on the repository or contact the maintainers.
