#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:29:38 2026

@author: YilberQuinto
"""

import os
import glob
import numpy as np
import librosa
import pywt
import cv2  
from concurrent.futures import ProcessPoolExecutor

def wdenoise_hard(data, wavelet='sym5', level=7):
    """
    Denoise (Universal Threshold, Hard, Level Dependent). 
    We apply the hard threshold to each level of detail individually.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        detail_coeffs = coeffs[i]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        new_coeffs.append(pywt.threshold(detail_coeffs, threshold, mode='hard'))
    
    return pywt.waverec(new_coeffs, wavelet)

def process_file(file_info):
    ruta_file, output_folder, filename, foldername = file_info
    
    try:
        # Audio loading (first channel automatically)
        datos, fs = librosa.load(ruta_file, sr=None)
        if datos.ndim > 1:
            datos = datos[0, :]
            
        # Central segment selection (2048 samples)
        N = len(datos)
        cent = N // 2
        datos_segmento = datos[cent - 1024 : cent + 1024]
        
        if len(datos_segmento) < 2048:
            return

        # Wavelet Denoising
        # datos_limpios = wdenoise_hard(datos_segmento, wavelet='sym5', level=7)

        # Mel-Espectrograma
        len_win = 256
        hop_length = int(len_win * 0.25)  # 75% overlap (OverlapLength = 0.75*128)
        n_fft = 2 * len_win               # 512
        
        # librosa calculates power by default instead of absolute magnitude
        s_mel = librosa.feature.melspectrogram(
            y=datos_segmento, 
            sr=fs, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=len_win, 
            window='hamming',
            center=False
        )
        
        # Convert to dB (+ eps to avoid log10(0))
        eps = np.finfo(float).eps
        s_dB = 10 * np.log10(np.abs(s_mel) + eps)
        
        # Normalize to [0, 1] 
        s_min, s_max = s_dB.min(), s_dB.max()
        if s_max - s_min > 0:
            s_norm = (s_dB - s_min) / (s_max - s_min)
        else:
            s_norm = np.zeros_like(s_dB)
            
        # Convert to uint8 -> equivalent to im2uint8
        s_img = (s_norm * 255).astype(np.uint8)
        s_img = np.flipud(s_img)
        
        # We use INTER_CUBIC to maintain maximum fidelity of the textures
        s_img_resized = cv2.resize(s_img, (80, 80), interpolation=cv2.INTER_CUBIC)
        
        # Grayscale -> RGB (Generates 3 identical channels)
        s_rgb = cv2.cvtColor(s_img_resized, cv2.COLOR_GRAY2BGR)
        
        # Save image directly without opening graphical windows
        namefig = os.path.join(output_folder, f"{foldername}_{filename.replace('.wav', '.wav.png')}")
        cv2.imwrite(namefig, s_rgb)
        
    except Exception as e:
        print(f"Error procesando {filename}: {e}")

def main():
    folders = ["aedes_male", "aedes_female", "fuit_flies", "house_flies",
               "quinx_male", "quinx_female", "stigma_male", "stigma_female",
               "tarsalis_male", "tarsalis_female"]
    
    input_path = '/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/'
    
    for foldername in folders:
        folder_path = os.path.join(input_path, foldername)
        
        # Ouput folder
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            
        files = glob.glob(os.path.join(folder_path, '*.wav'))
        tasks = [(f, foldername, os.path.basename(f), foldername) for f in files]
        
        print(f"Procesando {len(tasks)} archivos en paralelo para la carpeta: {foldername}...")
        
        # Parallel execution on 10 P-Cores
        with ProcessPoolExecutor(max_workers=10) as executor:
            executor.map(process_file, tasks)

if __name__ == "__main__":
    main()