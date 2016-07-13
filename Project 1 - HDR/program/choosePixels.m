% Choose pixels to recover camera response curve 
% Note: spatially well distributed in the image
% Input
%       imgs: the different exposure images
%       iStop: choose iStop rows
%       jStop: choose jStop cols
% Ouput
%       Zij: the choosen pixels
%   
function [Zij] = choosePixels(imgs,iStop,jStop)
    
    [h, w, channels, numOfImgs] = size(imgs);
    Zij = zeros(iStop*jStop, channels, numOfImgs, 'uint8');
    
    count = 1; % to count how many pixels have been chosen
    for i=1:iStop
        iChoice = round(h/iStop*(i-0.5)); 
        for j=1:jStop
            jChoice = round(w/jStop*(j-0.5));  
            for k=1:numOfImgs
                   for c=1:channels
                        Zij(count,c,k) = imgs(iChoice,jChoice,c,k);
                   end
            end
            count = count+1;
        end
    end
end

