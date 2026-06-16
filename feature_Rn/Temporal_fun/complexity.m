function y = complexity(x)

%%%%% Centroid %%%%%%%%%
% x - Vector  
 
N = length(x);

suma = 0;
for i = 1:N-1
    suma = suma + (x(i) - x(i+1)).^2;
end 
y = sqrt(suma);
end 