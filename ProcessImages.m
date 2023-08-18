#used to implement the MUSIC algorithm and extract an estimate of layer counts;

clear all;
close all;
% all_files = dir;
% all_files(1:2,:) = [];
% all_files(end,:) = [];

tab = readtable('woodtestimagelist.csv');
% file names
fn = tab(:,2);

heights = table2array(tab(:,4));
widths = table2array(tab(:,5));
laycount = table2array(tab(:,7));
indstart = table2array(tab(:,8));
indend = table2array(tab(:,9));


allwidths = 21;
maxNumIms = ceil(sum(widths)/allwidths);
BigTable = cell(maxNumIms,2);


tabind = 1;

for i = 1:22
    prefix = convertStringsToChars(string(fn{i,:}));
    FullIm = imread([prefix,'.jpg']);
    % getting rid of the DSC_ prefix
    prefix(1:4) = [];
    % making sure the image height is standard and 4000 pixels
    if size(FullIm,1) ~= 3072
        FullIm = imresize(FullIm,[3072,widths(i)]);
    end
    
    strip_start_ind = 1;
    strip_end_ind = allwidths;
    while 1
        strip = FullIm(:,strip_start_ind:strip_end_ind,:);
        if (strip_start_ind > indstart(i))&(strip_end_ind < (widths(i) - indend(i)))
            label = laycount(i);
        else
            strip_start_ind = strip_start_ind + allwidths;
            strip_end_ind = strip_end_ind + allwidths;
            if strip_end_ind > widths(i)
                break;
            end
            continue;
        end
        % writing the image and saving its label
        imwrite(strip,strcat('Strips/', prefix,'_',num2str(tabind),'.jpg'));
        BigTable{tabind,1} = strcat(prefix,'_',num2str(tabind),'.jpg');
        BigTable{tabind,2} = label;
        tabind = tabind+1;
        
        strip_start_ind = strip_start_ind + allwidths;
        strip_end_ind = strip_end_ind + allwidths;
        
        if strip_end_ind > widths(i)
            break;
        end
        
        
    end
end

BigTable((tabind+1):end,:) = [];
T = cell2table(BigTable,'VariableNames',{'FileName' 'Label'});
writetable(T, 'Test&ValidationDataLabels.csv');





