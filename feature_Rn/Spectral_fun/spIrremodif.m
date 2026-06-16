function y = spIrremodif(ampwk)

%%%%% Modified spectral irregularity %%%%%%%%%
% ampwk - vector de magnitudes de fft

N = length(ampwk);

suma1 = 0;
for k = 2:N
    S = (ampwk(k)-ampwk(k-1))^2;
    suma1 = suma1 + S;
end 

suma2 = sum(ampwk.^2);

y = suma1/suma2; 
end 

