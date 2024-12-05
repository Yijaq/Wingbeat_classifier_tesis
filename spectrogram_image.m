clear, clc, close all

folders = ["aedes_male", "aedes_female", "fuit_flies","house_flies",...
           "quinx_male", "quinx_female","stigma_male", "stigma_female",...
           "tarsalis_male", "tarsalis_female"]; 

for i = 1:length(folders)
    foldername = folders(i);
    ruta = '/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/';
    ruta_folder = strcat(ruta,foldername,'/');
    listdir = dir(fullfile(ruta_folder,'*.wav'));

    for k = 1:length(listdir)
        filename = listdir(k).name;              
        ruta_file = strcat(ruta_folder,filename);

        % Carga de archivo de audio 
        [datos, fs] = audioread(ruta_file);
        datos = datos(:,1);
        N = length(datos);
        cent = N/2; 
        liminf = cent + 0.5 - 1024; 
        limsup = cent - 0.5 + 1024;
        datos = datos(liminf:limsup);

        % Eliminacion de ruido 
        datos = wdenoise(datos,7, ...
                        Wavelet='sym5', ...
                        DenoisingMethod='UniversalThreshold', ...
                        ThresholdRule='Hard', ...
                        NoiseEstimate='LevelDependent');
    
        % Parámetros espetrograma 
        len_win = 128;                         
        win = hamming(len_win);           
        overlap = 0.75*len_win;           
        nfft = 2*len_win;                 
        [s, f, t] = spectrogram(datos,win,overlap,nfft,fs,'yaxis');
        s_dB = 20*log10(abs(s));

        % Grafica de espectrograma
        imagesc(t,f,s_dB);
        imaSize = [80, 80];
        set(gca,'Position',[0 0 1 1]);
        set(gca,'YDir','normal');
        set(gcf,'Position',[0, 0, imaSize]);
        xticks([]);
        yticks([]);
        colormap(gray)
        axis off;     
        namefig = sprintf('%s_%s.png',foldername,filename);
        print(namefig, '-dpng', '-r0');  % -r0 garantiza que no haya escalado de la imagen 
    end

    % Guardar espectrogramas en una carpeta
    foldername_destino = foldername;
    if ~exist(foldername_destino, 'dir')
        mkdir(foldername_destino)
    end
    movefile('*wav.png',foldername_destino);
end 