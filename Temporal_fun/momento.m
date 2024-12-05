% Momentun de una señal 

function y = momento(x,n)
% x es un vector 
% n es un Numero natural

N = length(x);
suma = 0;

for k = 1:N
    y = (x(k) - mean(x)).^n;
    suma = suma + y;
end 

y = suma./N; 
y = y./(std(x).^n);

end 