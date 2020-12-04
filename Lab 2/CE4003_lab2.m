clc;
clear;
close all;

%% Edge Detection

mac = imread('maccropped.jpg');

mac_g=rgb2gray(mac);

sob_x = double([-1 0 1;-2 0 2;-1 0 1]);
sob_y = sob_x';

Gx = conv2(mac_g,sob_x,'same');
Gy = conv2(mac_g,sob_y,'same');
Gxy = sqrt(Gx.^2 + Gy.^2);
Gt=Gxy>200;

figure('name','Sobel filtering');
subplot(131),imshow(mac_g),title('Original Mac');
subplot(132),imshow(Gx),title('Sobel (Gx)');
subplot(133),imshow(Gy),title('Sobel (Gy)')
figure('name','Sobel filtering w post processing');
subplot(121),imshow(Gxy),title('Sobel (x,y)')
subplot(122),imshow(Gt),title('Sobel (threshold)')

%Canny Edge Detection

for i = 1:5
    mac_canny1(:,:,i) = edge(mac_g,'canny',[0.04 0.1],i);
end

for j = 1:5
    mac_canny2(:,:,j) = edge(mac_g,'canny',[(0.015*j) 0.1],1);
end

figure('name','Canny Edge Detection (Changing sigma)')
subplot(231),imshow(mac_g),title('Original Mac');
subplot(232),imshow(mac_canny1(:,:,1)),title('Canny Sig=1');
subplot(233),imshow(mac_canny1(:,:,2)),title('Canny Sig=2')
subplot(234),imshow(mac_canny1(:,:,3)),title('Canny Sig=3')
subplot(235),imshow(mac_canny1(:,:,4)),title('Canny Sig=4')
subplot(236),imshow(mac_canny1(:,:,5)),title('Canny Sig=5')

figure('name','Canny Edge Detection (Changing tl)')
subplot(231),imshow(mac_g),title('Original Mac');
subplot(232),imshow(mac_canny2(:,:,1)),title('Canny tl=0.015');
subplot(233),imshow(mac_canny2(:,:,2)),title('Canny tl=0.030')
subplot(234),imshow(mac_canny2(:,:,3)),title('Canny tl=0.045')
subplot(235),imshow(mac_canny2(:,:,4)),title('Canny t1=0.060')
subplot(236),imshow(mac_canny2(:,:,5)),title('Canny tl=0.075')

%% Line Finding using Hough Transform 

[H,theta,rho] = hough(mac_canny1(:,:,3));

figure('name','Picture in hough domain')
imshow(imadjust(rescale(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)

P = houghpeaks(H,1);

x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','black');

lines = houghlines(mac_canny1(:,:,1),theta,rho,P);

figure('name','Line Finding using Hough Transform'), imshow(mac_g), hold on

xy = [lines(1).point1; lines(1).point2];
plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','red');

% Plot beginnings and ends of lines
plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','green');
plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','green');

%% 3D Stereo

%Using disparity function
corrL = imread('corridorl.jpg');
corrR = imread('corridorr.jpg');

corrL_g=rgb2gray(corrL);
corrR_g=rgb2gray(corrR);

disparityRange = [0 16];
disparityMap = disparityBM(corrL_g,corrR_g,'DisparityRange',disparityRange,'UniquenessThreshold',0);

% figure('name','Disparity Map 1')
% imshow(disparityMap,disparityRange)

triL = imread('triclopsi2l.jpg');
triR = imread('triclopsi2r.jpg');

triL_g=rgb2gray(triL);
triR_g=rgb2gray(triR);

disparityRange2 = [0 16];
disparityMap2 = disparityBM(triL_g,triR_g,'DisparityRange',disparityRange2,'UniquenessThreshold',0);

% figure('name','Disparity Map 2')
% imshow(disparityMap2,disparityRange2)

figure('name','Disparity Map overall')
subplot(231),imshow(corrL_g),title('Corridor Original');
subplot(232),imshow(disparityMap,disparityRange),title('Corridor DM lib');
subplot(234),imshow(triL_g),title('Triclops Original')
subplot(235),imshow(disparityMap2,disparityRange2),title('Triclops DM lib')

%Manual try 2

n= 11;

res = map(corrL_g, corrR_g, n);

subplot(233),imshow(res, [-15 15]),title('Corridor DM manual')

res2 = map(triL_g, triR_g, n);

subplot(236),imshow(res2, [-15 15]),title('Triclops DM manual')
