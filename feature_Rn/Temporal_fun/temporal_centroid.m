function centroid = temporal_centroid(x)

%%%%% Centroid %%%%%%%%%
% x - Vector  
 

N = length(x);

suma = 0;
for k = 1:N
    suma = suma + k.*x(k);
end 

NM = suma;         % .* multiplica componete a componete 
DM = sum(x);

centroid = NM/DM;
end 