function [ result ] = alignment( imgs, imgnum )
%ALIGNMENT Summary of this function goes here
%   Detailed explanation goes here

    grays = grayImage(imgs, imgnum);

    median = zeros(imgnum, 'int32');
    bw = zeros(size(grays,1), size(grays,2), imgnum, 'uint8');

    for i=1 : imgnum
        [bw(:,:,i), median(i)] = trans2bw( grays(:,:,i) );
    end

    exclusion = false(size(grays,1), size(grays,2), size(grays,3));
    for i=1 : size(exclusion,3)
        exclusion(:,:,i) = findExclusion(grays(:,:,i), median(i), 3);
    end

    % imshow(exclusion);
    shift = zeros(size(exclusion,3), 2);
    for i=2 : size(exclusion,3)
        [x, y] = pyramid(bw(:,:,i-1), bw(:,:,i), exclusion(:,:,i-1));
        shift(i, :) = [x y];
    end
    for i=2 : size(exclusion,3)
        shift(i, :) = shift(i-1, :) + shift(i, :);
    end
    
    result = imgs;
    
    for i=2 : size(exclusion,3)
        result(:,:,:,i) = shiftImage3(imgs(:,:,:,i), shift(i,:));
    end

end



function [ grayimgs ] = grayImage( imgs, num )
%GRAYIMAGE Summary of this function goes here
%   Detailed explanation goes here

    temp = imgs(:,:,:,1);
    grayimgs = zeros(size(temp,1), size(temp,2), num, 'uint8');
    
    for i=1 : num       
        grayimgs(:,:,i) = rgb2gray(imgs(:,:,:,i));
    end

end



function [ exclusion ] = findExclusion( grayimg, median, n )
%FINDEXCLUSION Summary of this function goes here
%   Detailed explanation goes here
%   return exclusion matrix with "true, false"

    exclusion = false(size(grayimg,1), size(grayimg,2));
    
    for i=1 : size(exclusion,1)
        for j=1 : size(exclusion,2)
            value = int32(grayimg(i,j));
            if median-n <= value && value <= median+n
                exclusion(i,j) = false;
            else
                exclusion(i,j) = true;
            end
        end
    end

end



function [ x, y ] = pyramid( img1, img2, exclusion )
%PYRAMID Summary of this function goes here
%   Detailed explanation goes here

    if size(img1, 2) <= 100
        xs = 0;
        ys = 0;
    else
        [xs, ys] = pyramid( img1(1:2:end, 1:2:end), img2(1:2:end, 1:2:end), exclusion(1:2:end, 1:2:end) );
        xs = xs * 2;
        ys = ys * 2;
    end
    
    direct = [ -1,-1; -1,0; -1,1; 0,-1; 0,0; 0,1; 1,-1; 1,0; 1,1 ];
    currmin = 1e9;
    for k=1 : size(direct,1)
        
        shift = [xs + direct(k,1), ys + direct(k,2)];
        shiftimg = shiftImage(img2, shift);
        
        diff = xor(img1, shiftimg);
        diff = and(diff, exclusion);
        curr = sum(sum(diff));
        
        if curr < currmin
            x = direct(k, 1);
            y = direct(k, 2);
            currmin = curr;
        end
    end
    
    x = xs + x;
    y = ys + y;
%     keyboard;

end



function [ median ] = accumulate( grayimg )
%ACCUMULATE Summary of this function goes here
%   Detailed explanation goes here

    counting = zeros(260, 'int32');
    
    for i=1 : size(grayimg,1)
        for j=1 : size(grayimg,2)
            counting( grayimg(i,j)+1 ) = counting( grayimg(i,j)+1 ) + 1;
        end
    end
    
    median = int32(1);
    sum = counting(median);
    while sum < (size(grayimg,1) * size(grayimg,2) / 2)
        median = median + 1;
        sum = sum + counting(median);
    end

end



function [ grayimg, median ] = trans2bw( grayimg )
%TRANS2BW Summary of this function goes here
%   Detailed explanation goes here

    median = accumulate(grayimg);

    for i=1 : size(grayimg,1)
        for j=1 : size(grayimg,2)
            if grayimg(i,j) > median
                grayimg(i,j) = 255;
            else
                grayimg(i,j) = 0;
            end
        end
    end

end



function [ shiftimg ] = shiftImage( img, shift )
%SHIFTIMAGE Summary of this function goes here
%   Detailed explanation goes here

    shiftimg = circshift(img, shift);

    if shift(1) < 0
        for i=size(shiftimg,1)-shift(1)+1 : size(shiftimg,1)
            for j=1 : size(shiftimg,2)
                shiftimg(i,j) = img(i,j);
            end
        end
    else
        for i=1 : shift(1)
            for j=1 : size(shiftimg,2)
                shiftimg(i,j) = img(i,j);
            end
        end
    end

    if shift(2) < 0
        for i=1 : size(shiftimg,1);
            for j=size(shiftimg,2)-shift(2)+1 : size(shiftimg,2)
                shiftimg(i,j) = img(i,j);
            end
        end
    else
        for i=1 : size(shiftimg,1);
            for j=1 : shift(2)
                shiftimg(i,j) = img(i,j);
            end
        end
    end

end



function [ shiftimg ] = shiftImage3( img, shift )
%SHIFTIMAGE Summary of this function goes here
%   Detailed explanation goes here

    shiftimg = circshift(img, shift);

    if shift(1) < 0
        for i=size(shiftimg,1)-shift(1)+1 : size(shiftimg,1)
            for j=1 : size(shiftimg,2)
                shiftimg(i,j,:) = img(i,j,:);
            end
        end
    else
        for i=1 : shift(1)
            for j=1 : size(shiftimg,2)
                shiftimg(i,j,:) = img(i,j,:);
            end
        end
    end

    if shift(2) < 0
        for i=1 : size(shiftimg,1);
            for j=size(shiftimg,2)-shift(2)+1 : size(shiftimg,2)
                shiftimg(i,j,:) = img(i,j,:);
            end
        end
    else
        for i=1 : size(shiftimg,1);
            for j=1 : shift(2)
                shiftimg(i,j,:) = img(i,j,:);
            end
        end
    end

end

