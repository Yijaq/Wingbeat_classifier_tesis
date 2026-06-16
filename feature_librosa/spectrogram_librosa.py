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
    # Descomposición
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Estimación de umbral universal (Universal Threshold)
    # sigma = MAD / 0.6745
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # Aplicar umbral duro (Hard Thresholding)
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='hard') for c in coeffs[1:]]
    # Reconstrucción
    return pywt.waverec(new_coeffs, wavelet)

def process_file(file_info):
    ruta_file, foldername_destino, filename, foldername = file_info
    
    try:
        # Carga de audio (librosa normaliza a [-1, 1] por defecto)
        datos, fs = librosa.load(ruta_file, sr=None)
        
        # Selección de segmento central (2048 muestras)
        n = len(datos)
        cent = n // 2
        liminf = cent - 1024
        limsup = cent + 1024
        
        # Evitar errores si el audio es muy corto
        if liminf < 0 or limsup > n:
            return
            
        datos = datos[liminf:limsup]
        
        # Eliminación de ruido (Wavelet)
        #datos_denoised = wdenoise_hard(datos, wavelet='sym5', level=7)
        
        # Parámetros espectrograma
        len_win = 128
        hop_length = int(len_win * 0.25) # 75% overlap
        
        # Generar espectrograma (STFT)
        stft = librosa.stft(datos, n_fft=2*len_win, hop_length=hop_length, 
                            win_length=len_win, window='hamming', center=False)
        s_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Configuración de la figura para que sea exactamente 224x224 sin bordes
        # dpi * pulgadas = píxeles -> 100 * 2.24 = 224px
        fig = plt.figure(figsize=(0.80, 0.80), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Ocupar toda la figura
        ax.imshow(s_db, aspect='auto',cmap='gray', origin='lower')
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
        
        # Ejecución en paralelo (equivale a parfor)
        with ProcessPoolExecutor(max_workers=10) as executor:
            executor.map(process_file, tasks)

if __name__ == "__main__":
    main()