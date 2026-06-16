function centroid = spCentroid(f,y)

%%%%% Spectral Centroid %%%%%%%%%
% f - Vector de frecuencias 
% y - Vector de magnitudes de fft 

N = length(f);

suma = 0;
for k = 1:N
    suma = suma + f(k).*y(k);
end 

NM = suma;         % .* multiplica componete a componete 
DM = sum(y);

centroid = NM/DM;
end 