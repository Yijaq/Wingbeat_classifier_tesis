clear, clc, close all
  
folders = ["aedes_male", "aedes_female", "fuit_flies", "house_flies",... 
           "quinx_male", "quinx_female", "stigma_male", "stigma_female",...
           "tarsalis_male", "tarsalis_female"]; 

for i=1:length(folders)
    foldername = string(folders(i));
    ruta = '/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/';
    ruta_folder = strcat(ruta,foldername,'/');
    listdir = dir(fullfile(ruta_folder,'*.wav'));

    for k = 1:length(listdir)
        filename = listdir(k).name;              
        ruta_file = fullfile(ruta_folder,filename);

        % Carga de archivo de audio 
        [datos, fs] = audioread(ruta_file);
        datos = datos(:,1);        % Primer canal 
        N = length(datos);
        cent = N/2; 
        liminf = cent + 0.5 - 1024; 
        limsup = cent - 0.5 + 1024;
        datos = datos(liminf:limsup);        
        N = length(datos);

        % Eliminacion de ruido 
        datos = wdenoise(datos,7, ...
                        Wavelet='sym5', ...
                        DenoisingMethod='UniversalThreshold', ...
                        ThresholdRule='Hard', ...
                        NoiseEstimate='LevelDependent');

        % Transformada wavelet continua 
        [cfs,frq] = cwt(datos,'morse',fs);
        tiempo = (0:N-1)/fs;

        % Grafica de escalogram
        imagesc(tiempo,frq,abs(cfs));
        xticks([]);
        yticks([]); 
        axis off;
        colormap(gray);
        set(gca, 'Position', [0 0 1 1]);
        set(gca, 'YDir', 'normal');
        set(gcf, 'Position', [0, 0, 80, 80]);     
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