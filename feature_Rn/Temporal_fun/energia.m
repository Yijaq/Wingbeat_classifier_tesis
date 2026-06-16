% Energía de una señal 

function y = energia(x)
% x es un vector 

N = length(x);
suma = 0;
for k = 1:N
    y = abs(x(k))^2;
    suma = suma + y;
end 
y = suma./N; 
end 
