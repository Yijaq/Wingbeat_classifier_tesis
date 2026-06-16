function [ts1,ts2,ts3] = tristimulus(ampwk)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ampwk  -  vector de magnitudes de armónicos 

suma = sum(ampwk);

ts1 = ampwk(1)/suma;
ts2 = sum(ampwk(2:4))/suma;
ts3 = sum(ampwk(5:end))/suma;
end 