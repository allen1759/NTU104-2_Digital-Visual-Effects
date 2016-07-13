% 
% Tone mapping local operator based on the paper "Photographic Tone Reproduction
% for Digital Images" (Process: Lw -> Lm -> Ld).
% Some parameters were set the same as the parametrs in the result of the
% paper.
% Input:
%       Lw: the world luminance of the hdr result image
%       delta: a small value to avoid the singularity that occurs if black
%       pixels are present in the image (i.e. log(0))
%       a: high key or low key (how light or dark it is)
%       phi: sharpening parametr (=8.0)
%       threshold: threshold V to select the corresponding scale Smax (=0.05)
% Output:
%       Ld: the output LDR result image
%
function [ Ld ] = localPTR( Lw, delta, a, phi, threshold)

    Ld = zeros(size(Lw));
    row = size(Lw,1);
    col = size(Lw,2);
  
    N = row*col; % the total number of pixels in the image
    AvgLw = exp(sum(log(Lw(:)+delta))/N);
    Lm = (a/AvgLw)*Lw;
    
    
    % Construct 9 Gaussian filters and convolve the image luminance with
    % Gaussian filters. The filtering would be operate at x-axis and yaxis separately. 
    Vi = zeros(row,col,8);
    V = zeros(row,col,8);
    alpha = 1/(2*sqrt(2));
    for scale=1:9
        s = 1.6^(scale-1);
        sigma = alpha*s;
        kernelSize = ceil(3*sigma); %The kernelSize is usually 3 to 5 times as the sigma.
        gaussianX =  fspecial('gaussian', [kernelSize 1], sigma);
        gaussianY = fspecial('gaussian', [1 kernelSize], sigma);
        Vi(:,:,scale) = conv2(Lm, gaussianX, 'same');
        Vi(:,:,scale) = conv2(Vi(:,:,scale), gaussianY, 'same');
    end
    
    % Compute the center-surround function
    for i=1:8
        V(:,:,i) = (Vi(:,:,i) - Vi(:,:,i+1)) ./ ((2^phi)*a / (s^2) + Vi(:,:,i));    
    end
    
    % To choose the largest neighborhood around a pixel with fairly even
    % luminance.
    Smax=zeros(row,col); % the scales for each pixel
    for i=1:row
        for j=1:col
            for s=1:size(V,3)
                if abs(V(i,j,s))>threshold % must narrow the area (the difference between Vi and Vi+1 is too large i.e. Vi and Vi+1 aren't similar.)
                    if s==1
                        Smax(i,j)=s;
                    else
                        Smax(i,j)=s-1;
                        break;
                    end
                end
            end
        end
    end
    
    % find some areas' similar neighborhood are bigger than the default size
    idx = find(Smax == 0);
    Smax(idx) = 8;
    
    Lsmaxblur = zeros(row,col);
    for i=1:row
        for j=1:col
            Lsmaxblur(i,j) = Vi(i,j,Smax(i,j));
        end
    end
    % constitue the local dodging-and-burning operator
    Ld = Lm./(1+Lsmaxblur);
    
    % value must be between 0 to 1
    index = find(Ld>1);
    Ld(index) = 1;
end

