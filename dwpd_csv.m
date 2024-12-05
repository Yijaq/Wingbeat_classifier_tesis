clear, clc, close all

path = "/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/**/*.wav";
listdir = dir(path);
NN = length(listdir);

RMS_dwpd           = zeros(NN,1);
interval_dwpd      = zeros(NN,1);
energy_dwpd        = zeros(NN,1);  
ZCR_dwpd           = zeros(NN,1);
CE_dwpd            = zeros(NN,1);
momento_3dwpd      = zeros(NN,1);
momento_4dwpd      = zeros(NN,1);
mean_dwpd          = zeros(NN,1);
variance_dwpd      = zeros(NN,1);
std_dwpd           = zeros(NN,1);
kurtosis_dwpd      = zeros(NN,1);
skewness_dwpd      = zeros(NN,1);
%cD_dwpd           = cell(NN,1);
Path_name          = cell(NN,1);
insect             = cell(NN,1);
   
for k = 1:NN
    filename        = listdir(k).name; 
    path_folder     = listdir(k).folder;
    list_path_foder = split(path_folder,'/');
    foldername      = string(list_path_foder(end));
    insect{k}      = foldername;

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
   
    % Wavelet 
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
    
    %Arbol de coefientes (ver documentacion)
    wpt = wpdec(datos,level,wname);  
    terminal_nodes = leaves(wpt);
    % wpd = [];      % vector features 
    len = length(terminal_nodes);
    for i = (len/2):len
        node_index = terminal_nodes(i);
        vW = [vW, wpcoef(wpt,node_index)']; 
    end 

    % Features 
    %cD_wt1_{k}          = cD';
    peaks = abs(max(cA) - min(cA));
    interval_dwpd(k)     = peaks;
    RMS_dwpd(k)          = rms(vW);  
    energy_dwpd(k)       = energia(vW); 
    momento_3dwpd(k)     = momento(vW,3);
    momento_4dwpd(k)     = momento(vW,4);
    ZCR_dwpd(k)          = zerocrossrate(vW);
    CE_dwpd(k)           = complexity(vW);
    mean_dwpd(k)         = mean(vW);
    variance_dwpd(k)     = var(vW); 
    std_dwpd(k)          = std(vW);
    kurtosis_dwpd(k)     = kurtosis(vW);                   
    skewness_dwpd(k)     = skewness(vW);                         
end

TT = table(Path_name,RMS_dwpd,interval_dwpd,energy_dwpd,momento_3dwpd,momento_4dwpd,ZCR_dwpd, ...
           CE_dwpd,mean_dwpd,variance_dwpd,std_dwpd,kurtosis_dwpd,skewness_dwpd,insect);

%TT.Properties.RowNames = names;
name_csv = sprintf('dwpd_%s_%d.csv',wname,level);
writetable(TT,name_csv,'Delimiter',',')
% Guardar datos en una carpeta
foldername_destino = 'Resultados';
if ~exist(foldername_destino, 'dir')
    mkdir(foldername_destino)
end
status = movefile(name_csv,foldername_destino);