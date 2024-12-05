function y = spFlux(f)

%%%%% Spectral Flux %%%%%%%%%
% f - vector de magnitudes de fft

N = length(f);
q = 2;

suma = 0;
for k = 1:N-1
    flux = abs(f(k)-f(k+1))^q;
    suma = suma + flux;
end 

y = (suma)^(1/q); 
end 