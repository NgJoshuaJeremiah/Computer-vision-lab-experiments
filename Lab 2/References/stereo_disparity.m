% Written by Pratik Jain
% Subscribe me on YouTube
% https://www.youtube.com/PratikJainTutorials
%%
clc 
clear 
% close all

%% Reading both the images
a = imread('triclopsi2l.jpg');     % left image
a1 = imread('triclopsi2r.jpg');    % right image

%% Filter size and converting images to double
n= 15;
n1 = ceil(n/2);
a = double(a);
a1 = double(a1);
b = size(a);

%% Defining how many pixels to search 
ser = 20;

% waitbar
msg = 'Creating Disparity map ...';
xw = 0;
fw = waitbar(xw,msg);
%% Main Loop for search
for i=n1:b(1)-n1                                                 %i scans rows of image a
    for j=n1:b(2)-n1                                             %j scans columns of image a 
        c=a(i-n1+1:i+n1-1,j-n1+1:j+n1-1,1:3);                        %putting the pixels inside the filter to a vector c        end
        for j1 = j:min([j+ser,b(2)-n1])                          %Scans the rows of image a1 to find disparity
            c1=a1(i-n1+1:i+n1-1,j1-n1+1:j1+n1-1,1:3);                %putting the pixels inside the filter to a vector c1
            sub(j1-j+1) = sum(abs(c-c1),'all');                  %putting the sum of absolute difference of pixels in 
                                                                 %image a and a1 in vector sub 
        end
        [minsub,argsub] = min(sub);                      % taking armin
        out(i,j) = argsub;                               %output image is formed here
    end
    xw = i/(b(1)-n1);
    waitbar(xw,fw) 
end
close(fw)
out1 = out/ser;                                           %Normalizing the output image

%% Showing the output
figure;
subplot(1,2,1)
imshow(out1);                                            % Displaying the images
subplot(1,2,2)
imshow(uint8(a));
                      
                        
                 
    