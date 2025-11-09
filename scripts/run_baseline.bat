@echo off
REM Create results directories if not exist
if not exist results\baseline mkdir results\baseline
if not exist _data mkdir _data

REM Train baseline (clean)
python train.py --mode baseline --epochs 20 --batch_size 128 --data_root ./_data

REM Evaluate baseline: clean + SNR curve
python evaluate.py --mode baseline --data_root ./_data --snr_list "35,30,25,20,15,10,5,0,-5"

echo Done. Results saved under results\baseline
pause
