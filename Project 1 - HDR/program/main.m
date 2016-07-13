% 
%   Inputs:
%       inputFolder
%       outputFolder
%       lambda: for recovering response curve
%       key: for tone mapping operation
%       Lwhite: for tone mapping operation - global operator
%       phi: for tone mapping operation - local operator
%       threshold: for tone mapping operation - local operator
%
function main(inputFolder,outputFolder,lambda,key,Lwhite,phi,threshold)
    
    % check input and output dictionary
    inputFolder = ['../images/',inputFolder];
    outputFolder = ['../result/',outputFolder];
    if(~exist(inputFolder))
        disp('Input failed');
        return
    end
    if(~exist(outputFolder))
        mkdir(outputFolder);
    end
    
        
    % loading images with different exposures
    [imgs,ln_Tj] = readImages(inputFolder);
    numOfImgs = size(imgs,4);
    disp('Finish loading images!');

    % images alignment
    imgs = alignment(imgs,numOfImgs);
    disp('Finish doing images aligment!');
    
    % construct Zij: choose 50 points (sampling) - N(P-1)>(Zmax-Zmin)
    Zij = choosePixels(imgs,5,10);
    disp('Finish choosing pixels!');

    %construct weigthing function
    w = weigthingFunc();

    % choose lambda for each color channel (smooth the response curve)
    % lambda = 22;

    % seperate R, G, B channels from images when calling gsolve
    ZijR = reshape(Zij(:,1,:), size(Zij,1), numOfImgs);
    [gR,lER] = gsolve(ZijR,ln_Tj,lambda,w);
    z = 1:256;
    subplot(2,2,1), plot(gR,z,'r');
    axis([-10,5, 0,260])
    xlabel('log exposure X');
    ylabel('pixel value Z');
    title('Red'); 
    ZijG = reshape(Zij(:,2,:), size(Zij,1), numOfImgs);
    [gG,lEG] = gsolve(ZijG,ln_Tj,lambda,w);
    subplot(2,2,2), plot(gG,z,'g');
    axis([-10,5, 0,260])
    xlabel('log exposure X');
    ylabel('pixel value Z');
    title('Green'); 
    ZijB = reshape(Zij(:,3,:), size(Zij,1), numOfImgs);
    [gB,lEB] = gsolve(ZijB,ln_Tj,lambda,w);
    subplot(2,2,3), plot(gB,z,'b');
    axis([-10,5, 0,260])
    xlabel('log exposure X');
    ylabel('pixel value Z');
    title('Blue'); 
    subplot(2,2,4), plot(gR,z,'r --', gG,z,'g', gB,z,'b -.');
    axis([-10,5, 0, 260]);   
    xlabel('log exposure X');
    ylabel('pixel value Z');
    title('Red(dashed), Green(solid), and Blue(dash-dotted) curves'); 
    saveas(gcf,[outputFolder,'/curve.jpg']); % gcf means the current figure handler
    disp('Finish recoverring response curve');

    % construct the HDR radiance map
    imgHDR = RadianceMap(imgs,gR,gG,gB,ln_Tj,w);
    disp('Finish constructing the HDR radiance map');

    % Write HDR file
    hdrname = [outputFolder,'/hdr.hdr'];
    hdrwrite(imgHDR,hdrname);

    % tone mapping
    disp('Using tonemap function of MATLAB');
    hdr = hdrread(hdrname);
    result = tonemap(hdr);
    imwrite(result,[outputFolder,'/tonemap.jpg']);

    Lw = getLuminance(imgHDR);
    %key = 0.3;
    delta = 0.00001;
    %Lwhite = 1.5;

    disp('Using global operator of Reinhard''s algorithm');
    LdGlobal = globalPTR(Lw,delta,key,Lwhite);
    globalResult = getRGBImage(imgHDR,Lw,LdGlobal);
    imwrite(globalResult,[outputFolder,'/global.jpg']);

    %phi = 8.0;
    %threshold = 0.05;
    disp('Using local operator of Reinhard''s algorithm');
    LdLocal = localPTR(Lw,delta,key,phi,threshold);
    localResult = getRGBImage(imgHDR,Lw,LdLocal);
    imwrite(localResult,[outputFolder,'/local.jpg']);
    
    disp('Done!');
end
