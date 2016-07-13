%
%   Get the world luminance of the hdr image
%   the formula is based on the results of the paper
%
function [ Lw ] = getLuminance( hdr )
     Lw = 0.27*hdr(:,:,1)+0.67*hdr(:,:,2)+0.06*hdr(:,:,3); 
end

