#provides the code to characterize and count the number of log cross-sections
%% File Names to Explore:
% 	test_242791_out.png				test_242807_out.png				test_242820_out.png
%   test_242793_out.png				test_242808_out.png				test_242824_out.png
% 	test_242797_out.png				test_242809_out.png				test_242827_out.png
% 	test_242798_out.png				test_242810_out.png				test_242831_out.png
% 	test_242799_out.png				test_242812_out.png				test_242832_out.png
% 	test_242800_out.png				test_242814_out.png				test_242835_out.png
% 	test_242801_out.png				test_242815_out.png				test_242837_out.png
% 	test_242802_out.png				test_242816_out.png				test_242838_out.png
% 	test_242804_out.png				test_242817_out.png				test_242839_out.png
% 	test_242805_out.png				test_242819_out.png				test_242841_out.png
%%
clear all;
clc;
img = imread('LogBinary/test_242814_out.png');
img = rgb2gray(img);
figure('position',[100 100 1600 600])
subplot(131);imshow(img);title('raw input image')
% Threshold and binarize image and fill holes 
binImg = imbinarize(img);
binImg = imfill(binImg, 'holes');

% Distance transform and watershed segmentation
D = bwdist(~binImg);
D = -D;

L = watershed(D);
% THE FOLLOWING IS ALL THE REASON WE APPLIED WATERSHED, TO SPLIT OVELAPPING
% REGIONS BY A BOUNDARY
L(~binImg) = 0;
% Generate label image
rgb = label2rgb(L,'jet',[.5 .5 .5]);
% Show results of watershed 

subplot(132);imshow(rgb)
title('Watershed transform of the image')


% Generate new binary image that only looks at individual regions generated
% by watershed segmentation so we are counting watershed regions, not label
% colors
binWatershed = L > 1; % 1 is background region; any region with index > 1 is log
minLogSize = 50; % minimum size in pixels
regs = regionprops(binWatershed, 'Area', 'Centroid', 'PixelIdxList');
% Remove all regions with size below threshold
regs(vertcat(regs.Area) < minLogSize) = [];


% Display image with coin count labeled
% These few lines are just for illustration purposes and mark a number on
% each log. We are done with the log counting at this point, and the number
% of logs is simply numel(regs)
numberOfLogs = numel(regs);

subplot(133);imshow(img);title(sprintf('labeled logs: total of %d logs', numberOfLogs ))
hold on
for k = 1:numel(regs)

    text(regs(k).Centroid(1), regs(k).Centroid(2), num2str(k), ...
        'Color', 'r', 'HorizontalAlignment', 'center')

end
hold off