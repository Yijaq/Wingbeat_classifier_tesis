function [flat,alpha] = flatness(ft)

%%%  Espectral flatness %%%
% ft - Vector magnitud de coeficientes de fft

flat = 10*log10(geomean(ft)/mean(ft));

a = flat/(-60);

alpha = min(a,1);
end 

