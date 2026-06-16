clear, clc, close all

path = "/Users/YilberQuinto/Desktop/Database/Classification_insects_Brasil/Dataset/**/*.wav";
list = dir(path);
NN = length(list);

w0         = zeros(NN,1);
ampw0      = zeros(NN,1);
inharm_    = zeros(NN,1); 
ts1        = zeros(NN,1);
ts2        = zeros(NN,1);
ts3        = zeros(NN,1);
SI         = zeros(NN,1);
SImod      = zeros(NN,1);
flux       = zeros(NN,1);
rolloff    = zeros(NN,1);
alpha      = zeros(NN,1);
centroid   = zeros(NN,1);
energy     = zeros(NN,1);
mean_      = zeros(NN,1);
var_       = zeros(NN,1);
std_       = zeros(NN,1);
kurtosis_  = zeros(NN,1);
skewness_  = zeros(NN,1);
Path_name  = cell(NN,1);
insect     = cell(NN,1);

for k = 1:NN
    filename        = list(k).name; 
    path_folder     = list(k).folder;
    list_path_foder = split(path_folder,'/');
    foldername      = string(list_path_foder(end));
    insect{k}       = foldername;
    NL =  strcat("Data/images/",foldername,"/",foldername,"_",filename,".png ",foldername);
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

    ft = fft(datos);                       
    f  = (0:length(ft)-1).*fs/length(ft);   
    y = abs(ft); 

    % Frecuencia fundamental 
    f0 = pitch(datos,fs,Range=[100,1600],WindowLength=2048,OverlapLength=0,Method="CEP");
    indx_f = find(f>=(f0-80) & f<=(f0+80));
    [M,I]  = max(y(indx_f));
    fw0    = f(I+indx_f(1)-1);  
    % El -1 es para cuadrar los indices

    len_win = 2048;
    [s,cf,t1] = melSpectrogram(datos,fs,...
                          'Window',hamming(len_win,"periodic"),...
                          'OverlapLength',0,...
                          'FFTLength',2*len_win,...
                          'FrequencyRange',[100,fs/2]);

    rolloff_  = spectralRolloffPoint(s,cf);
    flux_     = spectralFlux(datos,fs,Range=[100,fs/2]);
    
    w0(k)            = fw0; 
    ampw0(k)         = M;
    [ampwk,wk]       = armonic(y,f,fw0,7);
    inharm_(k)       = inharm(wk);     
    [ts1_,ts2_,ts3_] = tristimulus(ampwk);
    ts1(k)           = ts1_; 
    ts2(k)           = ts2_;
    ts3(k)           = ts3_;
    SI(k)            = spIrregularity(ampwk);   
    SImod(k)         = spIrremodif(ampwk);
    flux(k)          = energia(flux_);
    rolloff(k)       = rolloff_;
    [flat,alpha_]    = flatness(y);
    alpha(k)         = alpha_; 
    energy(k)        = energia(y);                                                                
    mean_(k)         = mean(y);
    var_(k)          = var(y); 
    std_(k)          = std(y);  
    kurtosis_(k)     = kurtosis(y);                   
    skewness_(k)     = skewness(y);                   
    centroid(k)      = spCentroid(f,y);            
end    

TT = table(Path_name,w0,ampw0,inharm_,ts1,ts2,ts3,SI,SImod,flux,...
           rolloff,alpha,energy,mean_,var_,std_,kurtosis_,skewness_,...
           centroid,insect);

name_csv = 'spectral.csv';
writetable(TT,name_csv,'Delimiter',',')
% Guardar datos en una carpeta
foldername_destino = sprintf('Resultados');
if ~exist(foldername_destino, 'dir')
    mkdir(foldername_destino)
end
status = movefile(name_csv,foldername_destino);