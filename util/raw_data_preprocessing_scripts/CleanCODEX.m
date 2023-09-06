clear all

RawDir = '/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Combplex/Experiments/CODEX/Combplex_stain5/Scan1/Extracted/';

MaskDir = '/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Combplex/Experiments/CODEX/Combplex_stain5/Scan1/Ilastik/inputs/';

test = 2; % 1=calibration, 0=apply aggregate removal to all FOVs
testFOV = 15; % FOV to calibrate on

ChannelNames = 'Tryptase'; % Name of channel to work on
% ChannelNames = {'CD8','CD45','HLA-DR_550','HLA-DR_647','Keratin_550','Keratin_750','Ki67','p53_647','p53_750','SMA_550','SMA_647','SMA_750'};
pool_size = [7 7]; % size of smoothing window
r = 50; % size of aggregates to remove

%%
filter = ones(pool_size) / prod(pool_size);

files = dir(RawDir);
dirFlags = [files.isdir];
subFolders = files(dirFlags);
subFolderNames = {subFolders(3:end).name};


%% 
% if test==1
%     for i=1:1 %Test aggregate removal size on 1 FOV and displays before and after
%         CurrMask = imread([MaskDir,char(subFolderNames(testFOV)),'_',char(ChannelNames),'.tif']);
%         CurrRaw = imread([MaskDir,char(subFolderNames(testFOV)),'_',char(ChannelNames),'.tif']);
%         % A = size(CurrRaw);
%         tmpMask = zeros(size(CurrMask));
%         % tmpMask = imresize(tmpMask,A);
%         tmpMask = double(CurrMask);
%         % tmpMask(CurrMask(:,:)==2)=0;
%         % pooled_mask = imfilter(tmpMask, filter, 'replicate');
%         pooled_mask = imbinarize(tmpMask);
%         % bw_mask_cleaned = bwareaopen(pooled_mask, r);
%         % CleanMask = tmpMask;
%         % CleanMask(bw_mask_cleaned(:,:)==0)=0    ;
%         CleanImg = CurrRaw;
%         CleanImg(pooled_mask(:,:)==0)=0;
%         % Display the results
%         % figure; plotbrowser on;
%         % subplot(1,2,1), imagesc(tmpMask), title('Original mask'); caxis([0 1]);
%         % subplot(1,2,2), imagesc(bw_mask_cleaned), title('Cleaned mask'); caxis([0 1]);
%         % 
%         figure; plotbrowser on;
%         subplot(1,2,1), imagesc(CurrRaw), title('Original Image'); caxis([0 5000]);
%         subplot(1,2,2), imagesc(CleanImg), title('Cleaned Image'); caxis([0 5000]);
%         % imwrite(CleanImg,fullfile(MaskDir,strcat(subFolderNames(57),"_",ChannelNames,"_Clean.tif")));
%     end
% else
%     for i=1:length(subFolderNames) %Run aggregate removal on all FOVs and save clean images
%         CurrMask = imread([MaskDir,char(subFolderNames(i)),'_',char(ChannelNames),'_550_Clean.tif']);
%         CurrRaw = imread([MaskDir,char(subFolderNames(i)),'_',char(ChannelNames),'_750_Raw.tif']);
%         % A = size(CurrRaw);
%         tmpMask = zeros(size(CurrMask));
%         tmpMask = double(CurrMask);
%         % tmpMask = imresize(tmpMask,A);
%         % tmpMask(CurrMask(:,:)==2)=0;
%         % pooled_mask = imfilter(tmpMask, filter, 'replicate');
%         pooled_mask = imbinarize(tmpMask);
%         % bw_mask_cleaned = bwareaopen(pooled_mask, r);
%         CleanImg = zeros(size(CurrRaw));
%         CleanImg = CurrRaw;
%         CleanImg(pooled_mask(:)==0)=0;
%         imwrite(CleanImg,fullfile(MaskDir,strcat(subFolderNames(i),"_",ChannelNames,"_750_Clean.tif")));
%     end
% end


% 
if test==1
    for i=1:1 %Test aggregate removal size on 1 FOV and displays before and after
        CurrMask = imread([MaskDir,char(subFolderNames(testFOV)),'-',char(ChannelNames),'_Simple Segmentation.tif']);
        CurrRaw = imread([MaskDir,char(subFolderNames(testFOV)),'-',char(ChannelNames),'.tif']);
        A = size(CurrRaw);
        tmpMask = zeros(size(CurrMask));
        tmpMask = imresize(tmpMask,A);
        tmpMask = double(CurrMask);
        tmpMask(CurrMask(:,:)==2)=0;
        pooled_mask = imfilter(tmpMask, filter, 'replicate');
        pooled_mask = imbinarize(pooled_mask);
        bw_mask_cleaned = bwareaopen(pooled_mask, r);
        CleanMask = tmpMask;
        CleanMask(bw_mask_cleaned(:,:)==0)=0    ;
        CleanImg = CurrRaw;
        CleanImg(bw_mask_cleaned(:,:)==0)=0;
        % Display the results
        figure; plotbrowser on;
        subplot(1,2,1), imagesc(tmpMask), title('Original mask'); caxis([0 1]);
        subplot(1,2,2), imagesc(bw_mask_cleaned), title('Cleaned mask'); caxis([0 1]);

        figure; plotbrowser on;
        subplot(1,2,1), imagesc(CurrRaw), title('Original Image'); caxis([0 1500]);
        subplot(1,2,2), imagesc(CleanImg), title('Cleaned Image'); caxis([0 1500]);
        % imwrite(CleanImg,fullfile(MaskDir,strcat(subFolderNames(57),"_",ChannelNames,"_Clean.tif")));
    end
else
    for i=1:length(subFolderNames) %Run aggregate removal on all FOVs and save clean images
        CurrMask = imread([MaskDir,char(subFolderNames(i)),'-',char(ChannelNames),'_Simple Segmentation.tif']);
        CurrRaw = imread([MaskDir,char(subFolderNames(i)),'-',char(ChannelNames),'.tif']);
        A = size(CurrRaw);
        tmpMask = zeros(size(CurrMask));
        tmpMask = double(CurrMask);
        tmpMask = imresize(tmpMask,A);
        tmpMask(CurrMask(:,:)==2)=0;
        pooled_mask = imfilter(tmpMask, filter, 'replicate');
        pooled_mask = imbinarize(pooled_mask);
        bw_mask_cleaned = bwareaopen(pooled_mask, r);
        CleanImg = zeros(size(CurrRaw));
        CleanImg = CurrRaw;
        CleanImg(bw_mask_cleaned(:)==0)=0;
        imwrite(CleanImg,fullfile(MaskDir,strcat(subFolderNames(i),"_",ChannelNames,"_Clean.tif")));
    end
end







