# SRCNN Image Super-Resolution (UE24CS352A Mini-Project, Problem 4)

Reimplementation of SRCNN (Dong et al.) following Garber, Grossman and Johnson-Yu,
"Image Super-Resolution Via a Convolutional Neural Network" (CS229, 2020). 2x upscaling at 224x224,
compared against nearest, bilinear and bicubic interpolation using PSNR.

## Setup
    pip install -r requirements.txt
    wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
    unzip -q DIV2K_valid_HR.zip

## Train and evaluate (100 images, 60/20/20 split, seed 42, batch size 2)
    python srcnn.py train --data_dir DIV2K_valid_HR --out_dir results --epochs 100 --batch_size 2

Outputs in results/: best_srcnn.pt, results.json, curves.png, samples.png.

## Demo
    python srcnn.py demo --image DIV2K_valid_HR/0801.png --ckpt results/best_srcnn.pt

Writes results/demo_output.png (bicubic, SRCNN, target, with PSNR values).

## Notes
- Model: Conv 9x9 (3->64), ReLU, Conv 1x1 (64->32), ReLU, Conv 5x5 (32->3). MSE loss, Adam, lr 0.001.
- PSNR is computed per image on RGB in [0,1] (MAX = 1) and averaged over the test set.
- Only the paper's best configuration is used (no batch norm, tanh or regularization).

## Results (test set, 20 images, seed 42, mean PSNR in dB)
Nearest 26.17, bilinear 26.76, bicubic 27.86, SRCNN 28.48 (batch size 2, 100 epochs).
With batch size 16 the model was undertrained and reached only 26.87 dB (below bicubic).
