% 
% Tone mapping global operator based on the paper "Photographic Tone Reproduction
% for Digital Images" (Process: Lw -> Lm -> Ld)
% Input:
%       Lw: the world luminance of the hdr image
%       delta: a small value to avoid the singularity that occurs if black
%       pixels are present in the image (i.e. log(0))
%       a: high key or low key (how light or dark it is)
%       Lwhite: tha smallest luminance that will be mapped to pure white
% Output:
%       Ld: the output LDR result image
%
function [ Ld ] = globalPTR( Lw, delta, a, Lwhite)
  
    Ld = zeros(size(Lw));
    row = size(Lw,1);
    col = size(Lw,2);
  
    N = row*col; % the total number of pixels in the image
    AvgLw = exp(sum(log(Lw(:)+delta))/N);
    Lm = (a/AvgLw)*Lw;
    Ld = (Lm.*(1+Lm/(Lwhite*Lwhite)))./(1+Lm);
    
    % value must be between 0 to 1
    index = find(Ld>1);
    Ld(index) = 1;
end

