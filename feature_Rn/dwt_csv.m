clear, clc, close all

path = "/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/**/*.wav";
list = dir(path);
NN = length(list);

RMS_wt1           = zeros(NN,1);
interval_wt1      = zeros(NN,1);
energy_wt1        = zeros(NN,1);  
ZCR_wt1           = zeros(NN,1);
CE_wt1            = zeros(NN,1);
momento_3wt1      = zeros(NN,1);
momento_4wt1      = zeros(NN,1);
mean_wt1          = zeros(NN,1);
variance_wt1      = zeros(NN,1);
std_wt1           = zeros(NN,1);
kurtosis_wt1      = zeros(NN,1);
skewness_wt1      = zeros(NN,1);
%cD_wt1_           = cell(NN,1);
Path_name         = cell(NN,1);
insect            = cell(NN,1);

for k = 1:NN
    filename        = list(k).name; 
    path_folder     = list(k).folder;
    list_path_foder = split(path_folder,'/');
    foldername      = string(list_path_foder(end));
    insect{k}      = foldername;

    NL =  strcat("Data/images/",foldername,"/",foldername,"_",filename,".png");
    Path_name{k} = NL;

    %Carga de archivo de audio
    path_file = fullfile(path_folder,filename);
    [datos, fs] = audioread(path_file);
    %Obtener primer canal
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
   
    % Wavelet feature 
    level = 5;
    wname = 'db5';
    %Primer nivel de descomposicion
    [cA, cD] = dwt(datos,wname);  
    vW = cD';
    for i = 1:level-1   
        [cA, cD] = dwt(cA,wname);
        vW = [vW cD'];
    end 
    vW = [vW cA'];
    peaks = abs(max(cA) - min(cA));

    % Features 
    %cD_wt1_{k}         = cD';
    interval_wt1(k)     = peaks;
    RMS_wt1(k)          = rms(vW);  
    energy_wt1(k)       = energia(vW); 
    momento_3wt1(k)     = momento(vW,3);
    momento_4wt1(k)     = momento(vW,4);
    ZCR_wt1(k)          = zerocrossrate(vW);
    CE_wt1(k)           = complexity(vW);
    mean_wt1(k)         = mean(vW);
    variance_wt1(k)     = var(vW); 
    std_wt1(k)          = std(vW);
    kurtosis_wt1(k)     = kurtosis(vW);                   
    skewness_wt1(k)     = skewness(vW);                         
end

TT = table(Path_name,RMS_wt1,interval_wt1,energy_wt1,momento_3wt1,momento_4wt1,ZCR_wt1, ...
           CE_wt1,mean_wt1,variance_wt1,std_wt1,kurtosis_wt1,skewness_wt1,insect);

%TT.Properties.RowNames = names;
name_csv = sprintf('dwt_%s_%d.csv',wname,level);
writetable(TT,name_csv,'Delimiter',',')
% Guardar datos en una carpeta
foldername_destino = 'Resultados';
if ~exist(foldername_destino, 'dir')
    mkdir(foldername_destino)
end
status = movefile(name_csv,foldername_destino);