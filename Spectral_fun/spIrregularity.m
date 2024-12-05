function y = spIrregularity(ampwk)

%%%%%%%%% Spectral irregularity   %%%%%%%%%%%
%
% ampwk -  Vector de magnitudes de los armonicos de una señal

N = length(ampwk);
Y = 20.*log10(ampwk);

suma = 0;
for k = 2:N-1
    EV = (Y(k-1) + Y(k) + Y(k+1))/3;
    SI = abs(Y(k) - EV);
    suma = suma + SI;
end 

y = suma; 
end 


