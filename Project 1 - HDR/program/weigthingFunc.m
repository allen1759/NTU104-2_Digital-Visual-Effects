%
% Construct weighting hat function
% No Inputs
% Output: 
%       w: the weighting function
%
function [ w ] = weigthingFunc()
    Zmin = 0; 
    Zmax = 255; 
    Zmid = (Zmax+Zmin+1)/2;
    w = zeros(1,256);
   
    for i=1:size(w,2)
        if i <= Zmid
            w(i) = (i-1)-Zmin;
        else
            w(i) = Zmax-(i-1);
        end
    end
end

