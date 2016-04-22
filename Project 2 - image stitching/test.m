Ia = imread('1.jpg');
Ib = imread('2.jpg');
Ia = single(rgb2gray(Ia)) ;
Ib = single(rgb2gray(Ib)) ;
[fa, da] = vl_sift(Ia) ;
[fb, db] = vl_sift(Ib) ;
[matches, scores] = vl_ubcmatch(da, db) ;


imgs = [imread('1.jpg');imread('2.jpg')];
height = size(imgs, 1)/2;
imshow(imgs);
hold on;
for i = 1 : size(matches,2)/5
    p1 = fa(1:2, matches(1,i));
    p2 = fb(1:2, matches(2,i)) + [0; height];
    plot( [p1(1) p2(1)], [p1(2) p2(2)], 'b');
end