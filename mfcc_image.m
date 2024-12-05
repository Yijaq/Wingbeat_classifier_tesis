clear, clc, close all

folders = ["aedes_male","aedes_female","fuit_flies","house_flies",...
           "quinx_male","quinx_female","stigma_male","stigma_female",...
           "tarsalis_male","tarsalis_female"]; 
 
for i=1:length(folders)
    foldername = char(folders(i));
    ruta = '/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/';
    ruta_folder = strcat(ruta,foldername,'/');

    lisdir = dir(fullfile(ruta_folder,'*.wav'));
    for k = 1:length(lisdir)
        filename = lisdir(k).name;     
        ruta_file = strcat(ruta_folder,filename);

        % Carga de archivo de audio 
        [datos, fs] = audioread(ruta_file);
        datos = datos(:,1);        
        cent = length(datos)/2; 
        liminf = cent + 0.5 - 1024; 
        limsup = cent - 0.5 + 1024;
        datos = datos(liminf:limsup); 

        % Eliminacion de ruido 
        datos = wdenoise(datos,7, ...
                     Wavelet='sym5', ...
                     DenoisingMethod='UniversalThreshold', ...
                     ThresholdRule='Hard', ...
                     NoiseEstimate='LevelDependent');

        % Transformada corta de Fourier 
        len_win = 128;                         
        win = hann(len_win,"periodic");                 
        overlap = 0.5*len_win;                
        nfft = 2*len_win;                    
        [s, f, t] = spectrogram(datos,win,overlap,nfft,fs,'yaxis');

        % Coeficientes cepstrales 
        try 
            mfcc(s,fs,"LogEnergy","Ignore");
        catch
            continue
        end
        imaSize = [80, 80];
        set(gca, 'Position', [0 0 1 1]);
        set(gca, 'YDir', 'normal');
        set(gcf, 'Position', [0, 0, imaSize]);   
        xticks([]);
        yticks([]);
        colormap(gray);
        colorbar off
        axis off;            
        namefig = sprintf('%s_%s.png',foldername,filename);
        print(namefig, '-dpng', '-r0');  % -r0 garantiza que no haya escalado de la imagen
    end

    % Guardar espectrogramas en una carpeta
    foldername_destino = sprintf('cep_imag_%s',foldername);
    if ~exist(foldername_destino, 'dir')
        mkdir(foldername_destino)
    end
    movefile('*wav.png',foldername_destino);
end