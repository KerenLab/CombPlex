function CODEXSaveTifs (savePath, dataMat, ChannelNames)
% function MibiSaveTifs (folder, dataMat, chanelNames)
% function gets 3d Mibi data and saves it as individual tif files

if 7~=exist(savePath,'dir')
    mkdir(savePath);
end

% Get the dimensions of the data
[rows, cols, numChannels] = size(dataMat);

% Define the desired size for the output images
outputRows = 2962; % Specify your desired height
outputCols = 2962; % Specify your desired width

for i=1:length(ChannelNames)
    data = uint16(dataMat(:,:,i));
    % Resize the data using imresize
    resizedData = imresize(data, [outputRows, outputCols]);
    imwrite(resizedData,[savePath,'/',ChannelNames{i},'.tif']);
end