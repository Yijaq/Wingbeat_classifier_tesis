#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:05:12 2026

@author: YilberQuinto
"""

import os
# NOTE: We have disabled ssqueezepy's internal parallelism.
# Since we will be using 10 cores with ProcessPoolExecutor to process separate files,
# this prevents threads from interfering with each other and overloading the CPU.
os.environ['SSQ_PARALLEL'] = '0'

import glob
import numpy as np
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Agg') # Set up matplotlib in non-graphical mode for background processes
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from ssqueezepy import cwt 

def procesar_archivo(task):
    filepath, filename, output_foldername, foldername = task
    try:
        # --- 1. Uploading an audio file ---
        fs, datos = wav.read(filepath)
        
        # If the audio is stereo, use the first channel
        if len(datos.shape) > 1:
            datos = datos[:, 0]
            
        # --- 2. Centering and cropping of 2048 samples ---
        N = len(datos)
        cent = N // 2
        liminf = cent - 1024
        limsup = cent + 1024
        datos = datos[liminf:limsup]
        
        if len(datos) < 2048:
            return  # Skip if the file is shorter than required
            
        # --- 3. Continuous Wavelet Transform (CWT) with ssqueezepy ---
        cfs, scales = cwt(datos, wavelet='gmw',fs=fs)
        
        # --- 4. Coefficient Homogenization (Logarithmic Scale in dB) ---
        magnitud = np.abs(cfs)
        # Convert to dB avoiding log(0)
        cfs_dB = 20 * np.log10(np.maximum(magnitud, 1e-5)) 
        # Normalize so that the maximum value is 0 dB
        cfs_dB = cfs_dB - np.max(cfs_dB) 
        # Apply dynamic range of -60 dB
        dynamic_range = -60
        cfs_homogenized = np.maximum(cfs_dB, dynamic_range)
        
        # --- 5. Figure HD_scalogram (80x80 pixels) ---
        fig = plt.figure(figsize=(0.80, 0.80), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) 
        ax.imshow(cfs_homogenized, aspect='auto', cmap='gray', origin='upper')
        ax.axis('off') 
        
        # Save image ensuring it is not re-scaled
        namefig = os.path.join(output_foldername, f"{foldername}_{filename.replace('.wav', '.wav.png')}")
        plt.savefig(namefig, dpi=100)
        plt.close(fig) 
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    # --- Path and folder configuration ---
    folders = ["aedes_male", "aedes_female", "fuit_flies", "house_flies",
               "quinx_male", "quinx_female", "stigma_male", "stigma_female",
               "tarsalis_male", "tarsalis_female"]
    
    input_path = '/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/'
    
    # Flattened task collection to maximize the use of all 10 P-Cores
    tasks = []
    for foldername in folders:
        ruta_folder = os.path.join(input_path, foldername)
        output_foldername = foldername
        
        os.makedirs(output_foldername, exist_ok=True)
        
        search_path = os.path.join(ruta_folder, '*.wav')
        listdir = glob.glob(search_path)
        
        for filepath in listdir:
            filename = os.path.basename(filepath)
            tasks.append((filepath, filename, output_foldername, foldername))
            
    # --- Parallel Execution ---
    print(f"Starting CWT processing with ssqueezepy for {len(tasks)} files...")
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(procesar_archivo, tasks)
        
    print("Process completed successfully!")