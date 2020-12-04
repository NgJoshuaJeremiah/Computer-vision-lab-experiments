clc
clear all
close all


%% Contrast Stretching

pc = imread('mrt-train.jpg'); 
%whos pc
p=rgb2gray(pc);
[m,n] = size(p);
figure('name','Contrast Stretching')
subplot(2,2,1),imshow(p), title('Original');
subplot(2,2,2),imhist(p), title('Original Histogram');

smallest = double(min(p(:)))/255;
largest = double(max(p(:)))/255;
s = imadjust(p,[smallest,largest],[]);

subplot(2,2,3),imshow(uint8(s)), title('Stretched');
subplot(2,2,4),imhist(uint8(s)), title('Stretched Histogram');


%% Histogram Equalisation

P3 = histeq(p,255);
P4 = histeq(P3,255);
figure('name','Histogram Equalisation')
subplot(3,2,1),imhist(uint8(p),10), title('Original, 10 bins');
subplot(3,2,2),imhist(uint8(p)), title('Original, 256 bins');
subplot(3,2,3),imhist(uint8(P3),10), title('Equalized, 10 bins');
subplot(3,2,4),imhist(uint8(P3)), title('Equalized, 256 bins');
subplot(3,2,5),imhist(uint8(P4),10), title('Equalized 2, 10 bins');
subplot(3,2,6),imhist(uint8(P4)), title('Equalized 2, 256 bins');


%% Gaussian Averaging Filters

gn=imread('ntu-gn.jpg');
sp=imread('ntu-sp.jpg');
[m1,n1] = size(gn);
[m2,n2] = size(sp);

sigma = 1.0;

kernel = zeros(5,5);        %for 5x5 kernel
W = 0;                      %sum of elements of kernel (for normalisation)
coeff = 1/(2*pi*sigma*sigma);
for i = 1:5
    for j =1:5
        sq_dist = (i-3)^2 + (j-3)^2;
        kernel(i,j) = coeff*exp(-1*(sq_dist)/(2*sigma*sigma));
        W = W + kernel(i,j);
    end
end
kernel = kernel/W;

output1 = zeros(m1,n1);
output2 = zeros(m2,n2);
gn1 = padarray(gn,[2,2]);
sp1 = padarray(sp,[2,2]);

for i=1:m1
    for j=1:n1
        temp = gn1(i:i+4 , j:j+4);
        temp = double(temp);
        conv =temp.*kernel;
        output1(i,j) = sum(conv(:));
    end
end

for i=1:m2
    for j=1:n2
        temp = sp1(i:i+4 , j:j+4);
        temp = double(temp);
        conv =temp.*kernel;
        output2(i,j) = sum(conv(:));
    end
end

output1 = uint8(output1);
output2 = uint8(output2);

figure('name','Guassian Filtering')
subplot(321),imshow(gn),title('Original Gn');
subplot(322),imshow(sp),title('Original Sp');
subplot(323),imshow(output1),title('Gaussian Filtered Gn, sigma=1')
subplot(324),imshow(output2),title('Gaussian Filtered Sp, sigma=1')


sigma = 2.0;                %for sigma = 2.0

kernel = zeros(5,5);        %for 5x5 kernel
W = 0;                      %sum of elements of kernel (for normalisation)
for i = 1:5
    for j =1:5
        sq_dist = (i-3)^2 + (j-3)^2;
        kernel(i,j) = coeff*exp(-1*(sq_dist)/(2*sigma*sigma));
        W = W + kernel(i,j);
    end
end
kernel = kernel/W;

output1 = zeros(m1,n1);
output2 = zeros(m2,n2);
gn1 = padarray(gn,[2,2]);
sp1 = padarray(sp,[2,2]);

for i=1:m1
    for j=1:n1
        temp = gn1(i:i+4 , j:j+4);
        temp = double(temp);
        conv =temp.*kernel;
        output1(i,j) = sum(conv(:));
    end
end

for i=1:m2
    for j=1:n2
        temp = sp1(i:i+4 , j:j+4);
        temp = double(temp);
        conv =temp.*kernel;
        output2(i,j) = sum(conv(:));
    end
end

output1 = uint8(output1);
output2 = uint8(output2);

subplot(325),imshow(output1),title('Gaussian Filtered Gn, sigma=2')
subplot(326),imshow(output2),title('Gaussian Filtered Sp, sigma=2')

figure('name','Kernel')
mesh(kernel);

%% median filtering

gn2 = padarray(gn,[1,1]);
sp2 = padarray(sp,[1,1]);

gn_median_3 = medfilt2(gn2,[3,3]);
sp_median_3 = medfilt2(sp2,[3,3]);

gn3 = padarray(gn,[2,2]);
sp3 = padarray(sp,[2,2]);

gn_median_5 = medfilt2(sp2,[5,5]);
sp_median_5 = medfilt2(sp2,[5,5]);

figure('name','Median Filtering')
subplot(321),imshow(gn),title('Original Gn');
subplot(322),imshow(sp),title('Original Sp');
subplot(323),imshow(gn_median_3),title('Median Filtered Gn, 3x3')
subplot(324),imshow(sp_median_3),title('Median Filtered Sp, 3x3')
subplot(325),imshow(gn_median_5),title('Median Filtered Gn, 5x5')
subplot(326),imshow(sp_median_5),title('Median Filtered Sp, 5x5')


%% Suppressing Noise Interference Patterns 1

pck=imread('pck-int.jpg');
[m3,n3] = size(pck);

F=fft2(pck);
S=log10(abs(fftshift(F)).^2);
S1=log10(abs(F).^2);

copy = zeros(m3,n3);

for i=1:m3
    for j=1:n3
        copy(i,j) = F(i,j);
        if (i>=238 && i<=242) && (j>=7 && j<=11)
           copy(i,j)=0;
        end
        if (i>=15 && i<=19) && (j>=247 && j<=251)
            copy(i,j)=0;
        end
    end
end

S2=log10(abs(fftshift(copy)).^2);
iF=uint8(real(ifft2(copy)));

figure('name', 'PCK TV');
subplot(221),imshow(pck), title('Original');
subplot(222),imshow(iF), title('Final');
subplot(223),imagesc(S), title('Power Spectrum');
subplot(224),imagesc(S2), title('Power Spectrum(Editted)');

%% Suppressing Noise Interference Patterns 2

prim=imread('primate-caged.jpg');
prim=rgb2gray(prim);
[m4,n4] = size(prim);

F1=fft2(prim);
G=log10(abs(fftshift(F1)).^2);
G1=log10(abs(F1).^2);

copy1 = zeros(m4,n4);

for i=1:m4
    for j=1:n4
        copy1(i,j) = F1(i,j);
        if (i>=251 && i<=256) && (j>=9 && j<=14)
           copy1(i,j)=0;
        end
        if (i>=5 && i<=10) && (j>=245 && j<=250)
            copy1(i,j)=0;
        end
    end
end

G2=log10(abs(fftshift(copy1)).^2);
iF2=uint8(real(ifft2(copy1)));

figure('name', 'Caged Primate');
subplot(221),imshow(prim), title('Original');
subplot(222),imshow(iF2), title('Final');
subplot(223),imagesc(G), title('Power Spectrum');
subplot(224),imagesc(G2), title('Power Spectrum(Editted)');


%% Undoing Perspective Distortion of Planar Surface 

img = im2double(rgb2gray(imread('book.jpg')));
msgid = 'Images:InitSize:adjustingMag'; %define warning msg
name = 'check2';

warning('off',msgid); %ignore warning message
figure('name', 'Image Transformation');
subplot(121),imshow(img),title('Original')

%[X Y] = ginput(4)
 
%column vector and row vectors for corners
c = [3 256 309 143]'; 
r = [160 216 46 28]'; 
base = [0 192; 167 192; 167 0; 0 0]; % fixed points

tf = fitgeotrans([c r],base,'projective'); %save geometric transformation based on vectors
disp('tf = ');
disp(tf)

T = tf.T; %transformation matrix
disp('T =');
format short g
disp(T);

%Overlay for original image to show red border and yellow labels
hold on;
plot([c;c(1)],[r;r(1)],'r','Linewidth',2);
text(c(1),r(1)+20,'0, 192','Color','y');
text(c(2)-40,r(2)+10,'167, 192','Color','y');
text(c(3)-50,r(3)-20,'167, 0','Color','y');
text(c(4),r(4)-20,'0, 0','Color','y');
hold off;

[xf1, xf1_ref] = imwarp(img,tf); %apply geomatric transform onto image
%subplot(122),imshow(xf1),title('Transformed Image')
xf1_ref

%Crop image
xf1_ref.XWorldLimits = [-10 175];
xf1_ref.YWorldLimits = [-10 200];
xf1_ref.ImageSize = [210 185];
[xf2 xf2_ref] = imwarp(img,tf,'OutputView',xf1_ref);
xf2_ref
subplot(122), imshow(xf2),title('Transformed Image');

