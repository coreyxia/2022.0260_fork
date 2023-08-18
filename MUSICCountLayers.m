#convert images into strips to be used for comparison between UNET and other machine learning approaches;

close all;
clear all;


b = double(rgb2gray(imread('IMG_0167_out.png')));


figure1 = figure;

for i = 1:size(b,2)    
    b(:,i) = b(:,i)-mean(b(:,i));
end

[sy,sx] = pmusic(b*b',4,[],size(b,1),'corr');

[~, thePeak] = max(sy);

axes = gca;
plot(sx,sy,'Parent',axes,'linewidth',2,'color',[0 0 0]);
[~, estNum] = max(sy);
hold on;
line([sx(estNum) sx(estNum)],[min(sy),1.01*max(sy)],'color',[ 0.85 0.33 0.1],'linewidth',2,'linestyle',':')
xlim([1,max(sx)])
ylim([min(sy),1.01*max(sy)]);
% Create ylabel
% Create ylabel



box(axes,'on');
set(axes,'FontSize',14,'Layer','top');
ylabel('$$S(f)$$','FontSize',24,'Interpreter','latex');

% Create xlabel
xlabel('$$f$$','FontSize',24,'Interpreter','latex');

