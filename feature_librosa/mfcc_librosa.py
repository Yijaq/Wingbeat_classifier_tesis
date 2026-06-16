#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:43:00 2026

@author: YilberQuinto
"""

import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pywt
from concurrent.futures import ProcessPoolExecutor

def wdenoise_hard(data, wavelet='sym5', level=7):
    """
    Equivalente aproximado a wdenoise de MATLAB con Hard Thresholding.
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='hard') for c in coeffs[1:]]
    return pywt.waverec(new_coeffs, wavelet)

def process_file(file_info):
    ruta_file, foldername_destino, filename, foldername = file_info
    
    try:
        # Carga de audio
        datos, fs = librosa.load(ruta_file, sr=None)
        
        # Selección de segmento central (2048 muestras)
        n = len(datos)
        cent = n // 2
        liminf = cent - 1024
        limsup = cent + 1024
        
        if liminf < 0 or limsup > n:
            return
            
        datos = datos[liminf:limsup]
        
        # Eliminación de ruido (Wavelet) 
        # datos = wdenoise_hard(datos, wavelet='sym5', level=7)
        
        # --- CÁLCULO DE MFCC ---
        # Definimos los parámetros de la ventana
        len_win = 256
        hop_length = int(len_win * 0.25) # 75% overlap (32 muestras)
        
        # Extraemos 40 coeficientes MFCC (puedes cambiar este número, ej: 13, 20, 40)
        # Usamos n_fft = 256 (2 * len_win) como tenías en tu STFT original
        mfccs = librosa.feature.mfcc(y=datos, sr=fs, n_mfcc=40, 
                                     n_fft=2*len_win, hop_length=hop_length, 
                                     win_length=len_win, window='hamming', center=False)
        
        # Configuración de la figura para que sea exactamente 224x224 píxeles sin bordes
        # dpi * pulgadas = píxeles -> 100 * 2.24 = 224px
        fig = plt.figure(figsize=(0.80, 0.80), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Ocupar toda la figura
        
        # Dibujar el MFCC
        ax.imshow(mfccs, aspect='auto',cmap='gray', origin='lower')
        ax.axis('off')
        
        # Guardar imagen
        namefig = os.path.join(foldername_destino, f"{foldername}_{filename.replace('.wav', '.wav.png')}")
        plt.savefig(namefig, dpi=100)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error procesando {filename}: {e}")

def main():
    folders = ["aedes_male", "aedes_female", "fuit_flies", "house_flies",
               "quinx_male", "quinx_female", "stigma_male", "stigma_female",
               "tarsalis_male", "tarsalis_female"]
    
    ruta_base = '/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/'
    
    for foldername in folders:
        ruta_folder = os.path.join(ruta_base, foldername)
        
        # Crear carpeta destino
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            
        # Listar archivos .wav
        files = glob.glob(os.path.join(ruta_folder, '*.wav'))
        
        # Preparar datos para el pool de procesos
        tasks = [(f, foldername, os.path.basename(f), foldername) for f in files]
        
        print(f"Procesando carpeta: {foldername} ({len(tasks)} archivos)...")
        
        # Ejecución en paralelo
        with ProcessPoolExecutor(max_workers=10) as executor:
            executor.map(process_file, tasks)

if __name__ == "__main__":
    main()