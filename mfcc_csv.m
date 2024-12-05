clear, clc, close all

path = "/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/**/*.wav";
list = dir(path);
NN = length(list);

mfcc_             = cell(NN,1);
Path_name         = cell(NN,1);
insect            = cell(NN,1);
    
for k = 1:NN
    filename        = list(k).name; 
    path_folder     = list(k).folder;
    list_path_foder = split(path_folder,'/');
    foldername      = string(list_path_foder(end));

    insect{k}   = foldername;
    NL =  strcat("Data/images/",foldername,"/",foldername,"_",filename,".png");
    Path_name{k} = NL;

    %Carga de archivo de audio
    path_file = fullfile(path_folder,filename);
    [datos, fs] = audioread(path_file);
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
    
    %Transformada corta de Fourier 
    len_win = 256;                          
    win = hamming(len_win,"periodic");
    overlap = 0.5*len_win;
    nfft = 2*len_win;                      
    [s, f, t] = spectrogram(datos,win,overlap,nfft,fs,'yaxis');

    % Coeficeintes cepstrales 
    try 
        coef_mfcc = mfcc(s,fs,"LogEnergy","Ignore");
    catch
        coef_mfcc = NaN(15,13);
    end     
    vect_mfcc = [coef_mfcc(:,1); coef_mfcc(:,2); coef_mfcc(:,3); coef_mfcc(:,4);...
                 coef_mfcc(:,5); coef_mfcc(:,6); coef_mfcc(:,7); coef_mfcc(:,8);...
                 coef_mfcc(:,9); coef_mfcc(:,10)];

    mfcc_{k} = vect_mfcc';
end    

TT = table(Path_name,mfcc_,insect);

name_csv = 'mfcc.csv';
writetable(TT,name_csv,'Delimiter',',')
% Guardar datos en una carpeta
foldername_destino = sprintf('Resultados');
if ~exist(foldername_destino, 'dir')
    mkdir(foldername_destino)
end
status = movefile(name_csv,foldername_destino);