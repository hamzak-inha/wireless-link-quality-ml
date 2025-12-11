# Wireless Link Quality Classification 

This project trains a simple machine learning model to predict wireless link quality (**good / medium / bad**) from basic link features.

## Dataset
`wireless_data.csv` includes:
- distance_m, tx_power_dBm, freq_GHz, walls, los, rssi_dBm
- throughput_Mbps (reference)
- quality_label (target)

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install pandas scikit-learn matplotlib
python train_wireless_quality.py
