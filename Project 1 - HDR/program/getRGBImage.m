%
%   Re-apply color channels
%
function [ result ] = getRGBImage( img, Lw, Ld )

    result = zeros(size(img,1),size(img,2),3);
    for channel=1:3
        temp = img(:,:,channel)./Lw;
        result(:,:,channel) = temp.*Ld;
    end

end

