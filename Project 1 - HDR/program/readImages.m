%
% Read  images with each exposure
% Input
%       folder: folder path containing images
% Output
%       imgs: [row, col, channel, i] i = # of imgs
%       Tj: exposureTime
%
function [ imgs,Tj ] = readImages(folder)
    files = dir([folder,'/*.', 'JPG']);
    %files = dir([folder,'/*.', 'png']);
    num = length(files); % get number of images in the folder
    file1name = [folder, '/', files(1).name];
    info = imfinfo(file1name);
    % initialize 
    imgs = zeros(info.Height, info.Width, 3, num, 'uint8');
   Tj = zeros(1,num);
    % Tj = log([1/0.03125 1/0.0625 1/0.125 1/0.25 1/0.5 1 1/2 1/4 1/8 1/16 1/32 1/64 1/128 1/256 1/512 1/1024]);
    for i=1:num
        fileName = [folder, '/',  files(i).name];
        imgs(:,:,:,i)=imread(fileName);
        info = imfinfo(fileName);
        Tj(i) = log(info.DigitalCamera.ExposureTime);
    end
end

