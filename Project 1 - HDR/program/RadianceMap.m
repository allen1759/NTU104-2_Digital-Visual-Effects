%
% Construct the HDR radiance map
% Input:
%       images: the original input images
%       gR, gG, gB: the camera response curve based on R, G, B, 3 channels
%       ln_Tj: log of exposure times of the input images
%       w: the weight hat function
% Output:
%       rmapHDR: the final HDR radiance map
%
function [ rmapHDR ] = RadianceMap(images, gR, gG, gB, ln_Tj, w )
    g = [gR gG gB];
    [height, width, channels, numOfImages] = size(images);
    lnE = zeros(height,width,channels);
    
    for c=1:channels
        for i=1:height
            for j=1:width
                total_lnE = 0;
                total_weight = 0;
                for n=1:numOfImages
                    intensity = images(i,j,c,n)+1;
                    total_lnE = total_lnE+w(intensity)*(g(intensity)-ln_Tj(n));
                    total_weight = total_weight+w(intensity);
                end
                lnE(i,j,c) = total_lnE/total_weight;
            end
        end
    end
    rmapHDR = exp(lnE);
    % remove NAN or INF
    index = find(isnan(rmapHDR) | isinf(rmapHDR));
    rmapHDR(index) = 0;
end

