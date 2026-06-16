clear, clc, close all

path = "/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/**/*.wav";
list = dir(path);
NN = length(list);

RMS                  = zeros(NN,1);
interval_t           = zeros(NN,1);
TC_t                 = zeros(NN,1);
ZCR_t                = zeros(NN,1);
CE_t                 = zeros(NN,1);
mean_t               = zeros(NN,1);
variance_t           = zeros(NN,1);
std_t                = zeros(NN,1);
kurtosis_t           = zeros(NN,1);
skewness_t           = zeros(NN,1);
energy_t             = zeros(NN,1);
Path_name            = cell(NN,1);
insect               = cell(NN,1);

for k = 1:NN
    filename        = list(k).name; 
    path_folder     = list(k).folder;
    list_path_foder = split(path_folder,'/');
    foldername      = string(list_path_foder(end));
    insect{k}       = foldername;

    % Ruta en Google Drive, nombre y etiquetas 
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

    %Features
    peaks = abs(max(datos) - min(datos));
    RMS(k)                 = rms(datos);     
    interval_t(k)          = peaks;
    TC_t(k)                = temporal_centroid(datos);
    ZCR_t(k)               = zerocrossrate(datos);
    CE_t(k)                = complexity(datos);
    mean_t(k)              = mean(datos);
    variance_t(k)          = var(datos); 
    std_t(k)               = std(datos);
    kurtosis_t(k)          = kurtosis(datos);                   
    skewness_t(k)          = skewness(datos);                    
    energy_t(k)            = energia(datos);         
end 

TT = table(Path_name,RMS,interval_t,TC_t,ZCR_t,CE_t,mean_t, ...
           variance_t,std_t,kurtosis_t,skewness_t, energy_t,insect);

name_csv = 'temporal.csv';
writetable(TT,name_csv,'Delimiter',',')
% Guardar datos en una carpeta
foldername_destino = sprintf('Resultados');
if ~exist(foldername_destino, 'dir')
    mkdir(foldername_destino)
end
status = movefile(name_csv,foldername_destino);