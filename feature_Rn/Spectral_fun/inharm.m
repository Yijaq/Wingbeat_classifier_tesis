function y = inharm(wk)

%%%%$$$ Inharmonicity %%%%%%%%%%%%%
% wk - vector de frecuencias de los armonicos  
%
% wk(1) es la frecuencia fundamental
%
N = length(wk);

suma = 0; 

for k = 1:N-1
    I = abs(wk(k+1) - (k+1)*wk(1))/((k+1)*wk(1));
    suma = suma + I;
end 
y = suma; 
end 