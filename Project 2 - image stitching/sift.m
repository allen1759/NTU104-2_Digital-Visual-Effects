function [  ] = sift( f1, f2 )
%SIFT Summary of this function goes here
%   Detailed explanation goes here

    file1 = [f1 '.jpg'];
    file2 = [f2 '.jpg'];
    
    Ia = imread(file1);
    Ib = imread(file2);
    Ia = single(rgb2gray(Ia)) ;
    Ib = single(rgb2gray(Ib)) ;
    [fa, da] = vl_sift(Ia) ;
    [fb, db] = vl_sift(Ib) ;
    [matches, scores] = vl_ubcmatch(da, db) ;


    imgs = [imread(file1);imread(file2)];
    height = size(imgs, 1)/2;
    imshow(imgs);
    hold on;
    for i = 1 : size(matches,2)/5
        p1 = fa(1:2, matches(1,i));
        p2 = fb(1:2, matches(2,i)) + [0; height];
        plot( [p1(1) p2(1)], [p1(2) p2(2)], 'b');
    end

    fa = fa';
    fb = fb';
    matches = matches';
    
    fd = fopen([f1 'fa.txt'], 'wt'); % Open for writing
    for i=1:size(fa,1)
       fprintf(fd, '%f %f %f %f', fa(i,:));
       fprintf(fd, '\n');
    end
    fclose(fd);
    
    fd = fopen([f1 'fb.txt'], 'wt'); % Open for writing
    for i=1:size(fb,1)
       fprintf(fd, '%f %f %f %f', fb(i,:));
       fprintf(fd, '\n');
    end
    fclose(fd);
    
    fd = fopen([f1 'ma.txt'], 'wt'); % Open for writing
    for i=1:size(matches,1)
       fprintf(fd, '%d %d', matches(i,:));
       fprintf(fd, '\n');
    end
    fclose(fd);
    
    
%     keyboard;
end

