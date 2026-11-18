@echo off
rem Train both seed-42 models one after the other (detached from the editor session).
cd /d "%~dp0.."
set DL_THREADS=12
python -u experiments\train_dl.py --model unet --seed 42 > results\dl\unet_seed42.out 2>&1
python -u experiments\train_dl.py --model attention_unet --seed 42 > results\dl\attention_unet_seed42.out 2>&1
echo done > results\dl\seed42_done.txt
