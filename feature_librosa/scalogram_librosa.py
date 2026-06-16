#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:05:32 2026

@author: YilberQuinto
"""

import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from ssqueezepy import cwt
from concurrent.futures import ProcessPoolExecutor

def process_file(file_info):
    ruta_file, foldername_destino, filename, foldername = file_info
    
    try:
        # 1. Carga de archivo de audio
        # sr=None mantiene la frecuencia de muestreo original
        datos, fs = librosa.load(ruta_file, sr=None)
        
        # Si es estéreo, tomar el primer canal (como datos(:,1) en MATLAB)
        if datos.ndim > 1:
            datos = datos[0, :]
            
        # 2. Selección de segmento central (2048 muestras)
        N_total = len(datos)
        cent = N_total // 2
        # El rango en Python es [inicio:fin], tomamos 1024 a cada lado
        datos_segmento = datos[cent - 1024 : cent + 1024]
        
        if len(datos_segmento) < 2048:
            return

        # 3. Transformada Wavelet Continua (CWT)
        # Usamos wavelet 'morse' para igualar a MATLAB
        # ssqueezepy devuelve (coeficientes, frecuencias)
        W, _ = cwt(datos_segmento, wavelet='morlet', fs=fs)
        escalograma = np.abs(W)

        # 4. Generación de imagen (224x224 píxeles)
        # 2.24 pulgadas * 100 DPI = 224 píxeles
        fig = plt.figure(figsize=(0.80, 0.80), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Ocupar toda la figura sin bordes
        
        # Graficar escalograma
        ax.imshow(escalograma, aspect='auto', origin='upper',cmap='jet')
        
        ax.axis('off') # Quitar ejes y ticks
        
        # 5. Guardar resultado
        namefig = os.path.join(foldername_destino, f"{foldername}_{filename.replace('.wav', '.wav.png')}")
        plt.savefig(namefig, dpi=100, pad_inches=0)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error procesando {filename}: {e}")

def main():
    folders = ["aedes_male", "aedes_female", "fuit_flies", "house_flies",
               "quinx_male", "quinx_female", "stigma_male", "stigma_female",
               "tarsalis_male", "tarsalis_female"]
    
    path_dataset = '/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/'
    
    for foldername in folders:
        path_folder = os.path.join(path_dataset, foldername)
        
        # Crear carpeta destino
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            
        # Listar archivos .wav
        files = glob.glob(os.path.join(path_folder, '*.wav'))
        
        # Preparar argumentos para el procesamiento paralelo
        tasks = [(f, foldername, os.path.basename(f), foldername) for f in files]
        
        print(f"Procesando {len(tasks)} archivos en: {foldername}...")
        
        # Ejecución en paralelo (equivalente a parfor con 10 núcleos)
        with ProcessPoolExecutor(max_workers=10) as executor:
            executor.map(process_file, tasks)

if __name__ == "__main__":
    main()