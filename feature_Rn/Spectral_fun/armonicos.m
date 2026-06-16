function [ampwk,wk] = armonic(y,f,w0,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entradas
% y - vector de magnitudes de fft  
% f - vector de frecuencias 
% w0 - frecuencia fundamental
% n - Número de armónicos análiszados
% 
% Salidas 
% wk  - frecuencia del k-ésimo armonico estimado  
% amp - magnitud del k-ésimo armonico estimado
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
ampwk = zeros(1,n);     % Magnitud de los armónicos 
wk    = zeros(1,n);       % Frecuecia de los armicos  

for k = 2:(n+1)
    indx_f = find(f>=(k*w0-80) & f<=(k*w0+80));
    [M,I] = max(y(indx_f));
    wk(k) = f(I+indx_f2(1)-1);
    ampwk(k) = M;
end 

end 
