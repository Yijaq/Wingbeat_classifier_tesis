clear, clc, close all

path = "/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/**/*.wav";
list = dir(path);
NN = length(list);

RMS_wpc               = zeros(NN,1);
interval_wpc          = zeros(NN,1);
energy_wpc            = zeros(NN,1);  
ZCR_wpc               = zeros(NN,1);
CE_wpc                = zeros(NN,1);
momento_3_wpc         = zeros(NN,1);
momento_4_wpc         = zeros(NN,1);
mean_wpc              = zeros(NN,1);
variance_wpc          = zeros(NN,1);
std_wpc               = zeros(NN,1);
kurtosis_wpc          = zeros(NN,1);
skewness_wpc          = zeros(NN,1);
Path_name             = cell(NN,1);
insect                = cell(NN,1);
   
for k = 1:NN
    filename        = list(k).name; 
    path_folder     = list(k).folder;
    list_path_foder = split(path_folder,'/');
    foldername      = string(list_path_foder(end));

    NL =  strcat("Data/images/",foldername,"/",foldername,"_",filename,".png");
    Path_name{k}   = NL;
    insect{k}      = foldername;

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
   
    % Transformada Wavelet Packets
    level = 5;
    wname = 'db5';
    %Arbol de coefientes (ver documentacion)
    wpt = wpdec(datos,level,wname);  
    terminal_nodes = leaves(wpt);
    wpd = [];      
    len = length(terminal_nodes);
    for i = 1:len
        node_index = terminal_nodes(i);
        %Cuefientes wavelet en el nodo indicado (ver wpt)
        wpd = [wpd, wpcoef(wpt,node_index)']; 
    end 

    % Features
    peaks             = abs(max(wpd) - min(wpd));
    interval_wpc(k)   = peaks;
    RMS_wpc(k)        = rms(wpd);  
    energy_wpc(k)     = energia(wpd); 
    momento_3_wpc(k)  = momento(wpd,3);
    momento_4_wpc(k)  = momento(wpd,4);
    ZCR_wpc(k)        = zerocrossrate(wpd);
    CE_wpc(k)         = complexity(wpd);
    mean_wpc(k)       = mean(wpd);
    variance_wpc(k)   = var(wpd); 
    std_wpc(k)        = std(wpd);
    kurtosis_wpc(k)   = kurtosis(wpd);                   
    skewness_wpc(k)   = skewness(wpd);
end

TT = table(Path_name,RMS_wpc,interval_wpc,energy_wpc,momento_3_wpc,momento_4_wpc,ZCR_wpc, ...
           CE_wpc,mean_wpc,variance_wpc,std_wpc,kurtosis_wpc,skewness_wpc,insect);

name_csv = sprintf('wpd_%s_%d.csv',wname,level);
writetable(TT,name_csv,'Delimiter',',');
% Guardar datos en una carpeta
foldername_destino = sprintf('Resultados');
if ~exist(foldername_destino, 'dir')
    mkdir(foldername_destino)
end
status = movefile(name_csv,foldername_destino);