% MIBIsaveFolderOfMultiTiffsAsFolder
% The function receives a folder of multiTIFFs as downloaded from the tracker and
% breaks each image to a folder of channel stacks

% enter the directory name of the multiTIFF folder
ProjectDir = '/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Combplex/Experiments/CODEX/Combplex_stain4_25AB/Scan1/QuPath/';
multiTifDir = fullfile(ProjectDir,'cores');
saveDir = fullfile(ProjectDir,'Extracted');
% multiTifDir = '/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Compressed sensing/Experiments/CODEX/3_BreastCA-TMA_stain2/alignment/cores/';
% saveDir = '/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Compressed sensing/Experiments/CODEX/3_BreastCA-TMA_stain2/alignment/Extracted/';

% ChannelNames = readtable(fullfile(ProjectDir,"MarkerList.txt"));
ChannelNames = ["DAPI","FOXP3","CD163","CD4","COL1A1","CD206","CD3","HLA-DR","CD8","HLA-1","CD45","CD15","Ki-67","CD68","CD14","CD38","CD20","CD31","SPARC",...
    "Tryptase","Na-K-ATPase","CK7","Pan-Keratin","SMA"];
% ChannelNames = ["DAPI","Keratin-Ki67-SMA-HLA","CD45","p53_647"];

% script
mkdir (saveDir);
mtFileNames = dir([multiTifDir,'/*.tif']);



for i = 1:length(mtFileNames)
    Filetif = mtFileNames(i).name;
    disp(['Working on image ',num2str(i) , ': ',Filetif]);
    [FinalImage,chanelNames] =  CODEXLoadqpTiff([multiTifDir,'/',Filetif]);

    % save single TIFs
    [pathstr, fname, ext] = fileparts(Filetif);
    CODEXSaveTifs ([saveDir,'/',fname,'/'], FinalImage, ChannelNames); %nofar added TIF
    %MibiSaveTifs ([tifdir,'/',fname,'/'], FinalImage, chanelNames);
end

Filetif = mtFileNames(107).name;
disp(['Working on image ',num2str(107) , ': ',Filetif]);
[FinalImage,chanelNames] =  CODEXLoadqpTiff([multiTifDir,'/',Filetif]);

% save single TIFs
[pathstr, fname, ext] = fileparts(Filetif);
CODEXSaveTifs ([saveDir,'/',fname,'/'], FinalImage, ChannelNames); %nofar addedd TIF
%MibiSaveTifs ([tifdir,'/',fname,'/'], FinalImage, chanelNames);