"""
Mira Welner
June 2025

This script loads Dr Dey's heartrate data and reports the heartrate (60k/rr interval) as well as the QT intervals.
It then creates a training table of length=5000 for each snip for both RR and QT. It is recorded at 20Hz so this represents
250 seconds, or about 4.5 minutes.

It then loads Dr Cong's ECG and PPG data and proccesses it similarly, with snips of 5000 datapoints each. However the ECG hz is
set at 500Hz, and so the PPG is upsampled to 500hz. This means that the snips are 10 seconds each.
"""

import numpy as np
from scipy.interpolate import interp1d
from glob import glob
import itertools
import matplotlib.pyplot as plt

ecg_hz = 500
heartrate_hz = 20
snip_len = 5000

def scale_data(x:np.ndarray) -> np.ndarray:
    x_scaled = (x - x.min()) / (x.max() - x.min())
    return x_scaled

def process_rr(rr_distance_ms):
    """
    "Given the ms distance between r peaks, return a numpy matrix
    of
    """
    bpm = 60000/rr_distance_ms
    scaled_heartrate = scale_data(bpm)
    num_samples = len(scaled_heartrate)//snip_len
    scaled_heartrate_trimmed = scaled_heartrate[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    return heartrate_snips

def process_qt(qt_distance_ms):
    scaled_qt = qt_distance_ms/350-1
    num_samples = len(scaled_qt)//snip_len
    scaled_heartrate_trimmed = scaled_qt[:num_samples*snip_len]
    heartrate_snips = scaled_heartrate_trimmed.reshape(num_samples, snip_len)
    return heartrate_snips

def proccess_heartrate_file(rr_distance_ms, qt_distance_ms,timestamps):
    qt_times_s = np.array([s / 1000 for s in itertools.accumulate(rr_distance_ms)], dtype=np.float64).flatten()
    f_interp_rr = interp1d(timestamps, rr_distance_ms, kind='linear')
    x_rr = np.arange(timestamps.min(), timestamps.max()-1, 1/heartrate_hz)
    interpolated_rr = f_interp_rr(x_rr)

    f_interp_qt = interp1d(qt_times_s, qt_distance_ms, kind='linear')
    x_qt = np.arange(qt_times_s.min(), qt_times_s.max()-1, 1/heartrate_hz)
    interpolated_qt = f_interp_qt(x_qt)
    return process_rr(interpolated_rr), process_qt(interpolated_qt)


all_ecg_data = np.array([])
all_ppg_data = np.array([])
for i in range(2,5): #no ppg data for patient 1
    ecg_filename = f'cong_ecg_ppg_data/ecg_signal_{i}.txt'
    ecg_file =np.loadtxt(ecg_filename)
    if i == 4:
        ecg_file = np.clip(ecg_file, 8150, 8300)
    ecg_file = scale_data(ecg_file)
    fig = plt.figure()
    plt.plot(ecg_file)
    plt.show()
    all_ecg_data = np.append(all_ecg_data,ecg_file)
    ppg_for_patient = np.array([])
    for j in range(1,6):
        ppg_filename = f'cong_ecg_ppg_data/ecg{i}_ppg_0{j}.txt'
        ppg_file =np.loadtxt(ppg_filename, skiprows=1)
        _, ppg_values = ppg_file.T
        ppg_for_patient = np.append(ppg_for_patient,ppg_values)
    interpolated_ppg_for_patient = np.interp(np.linspace(0,1,len(ecg_file)), np.linspace(0,1,len(ppg_for_patient)), ppg_for_patient)
    all_ppg_data = np.append(all_ppg_data,interpolated_ppg_for_patient)

num_ecg_samples = len(all_ecg_data)//snip_len
ecg_trimmed = all_ecg_data[:num_ecg_samples*snip_len]
ecg_snips = ecg_trimmed.reshape(num_ecg_samples, snip_len)
np.savetxt("processed_data/ecg.csv", ecg_snips, delimiter = ",")

num_ppg_samples = len(all_ppg_data)//snip_len
ppg_trimmed = all_ppg_data[:num_ppg_samples*snip_len]
ppg_snips = ppg_trimmed.reshape(num_ppg_samples, snip_len)
np.savetxt("processed_data/ppg.csv", ppg_snips, delimiter = ",")


rr = []
qt = []


for i in range(1,5):
    file_path = f"heartrate_data/18-58-25_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    data_file = np.genfromtxt(file_path, skip_header=1, usecols=(1, 3, 4), delimiter=',', filling_values=0.0, dtype=float)
    timestamps, rr_distance_ms, qt_distance_ms = data_file.T
    new_rr, new_qt = proccess_heartrate_file(rr_distance_ms,qt_distance_ms,timestamps)
    rr.append(new_rr)
    qt.append(new_qt)

for i in range(1,5):
    file_path = f"heartrate_data/07-15-37_8_hours_part{i}_v2_wholecaseRRiQTi.csv"
    data_file = np.genfromtxt(file_path, skip_header=1, usecols=(1, 3, 4), delimiter=',', filling_values=0.0, dtype=float)
    timestamps, rr_distance_ms, qt_distance_ms = data_file.T
    new_rr, new_qt = proccess_heartrate_file(rr_distance_ms,qt_distance_ms,timestamps)
    rr.append(new_rr)
    qt.append(new_qt)


#the below files are very short so they need to be concatinated before being processed
short_files_rr = []
short_files_qt = []
short_files_timestamps = []
for i in range(1,28):
    file_path = glob(f"heartrate_data/11-03-38_8_hours_part2_v2_everyRRQTinputIntoEntropy_Rel{i}_Abs*")[0]
    data_file = np.genfromtxt(file_path, skip_header=1, delimiter=',', filling_values=0.0, dtype=float)
    timestamps = data_file[:,0]
    if len(short_files_timestamps):
        timestamps+=short_files_timestamps[-1]
    rr_distance_s, qt_distance_s = data_file[:, [1, 2]].T
    short_files_rr.extend(rr_distance_s*1000)
    short_files_qt.extend(qt_distance_s*1000)
    short_files_timestamps.extend(timestamps)

new_rr, new_qt = proccess_heartrate_file(np.array(short_files_rr),np.array(short_files_qt),np.array(short_files_timestamps))
rr.append(new_rr)
qt.append(new_qt)

np.savetxt("processed_data/rr.csv", np.vstack(rr), delimiter = ",")
np.savetxt("processed_data/qt.csv", np.vstack(qt), delimiter = ",")
