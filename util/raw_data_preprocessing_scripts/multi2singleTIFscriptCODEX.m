% Multi to Single TIF for CODEX

% input multi-tif data should be in format: {main_dir}/{multi_dir}/{fov_name.tif}
% output single-tif data will be in format: {main_dir}/{single_dir}/{fov_name}/{channel.tif}


%% Set directory names and read channel_names text file

main_dir = "/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Compressed sensing/Experiments/CODEX/Datasets/Schuerch/MultiTumor";
multi_dir = fullfile(main_dir, "cores");
single_dir = fullfile(main_dir, "Extracted");
channel_names = readtable(fullfile(main_dir, "channelNamesMultiTumor.txt"), "Delimiter", ",", "TextType","string", "ReadVariableNames",false);

%% Channel Names

% format channel names table
channel_names.Properties.VariableNames = "channel";
channel_names.remove = zeros(height(channel_names),1);
% sets channels with blank or empty in them to be removed
channel_names.remove(contains(channel_names.channel, "blank")) = 1;
channel_names.remove(contains(channel_names.channel, "empty")) = 1;
% removes anything after " - " (Schurch)
channel_names.channel(contains(channel_names.channel, " - ")) = extractBefore(channel_names.channel(contains(channel_names.channel, " - ")), " - ");
% replaces " " and "-" with "_"
channel_names.channel = strrep(channel_names.channel, " ", "_");
channel_names.channel = strrep(channel_names.channel, "-", "_");

% you can also open channel names table and set remove column = 1 for blanks
% can also change channel names manually in the table.

%% Read FOVs

fov_nms = unique(string(struct2table(dir(fullfile(multi_dir, "*.tif*"))).name));
[~, fovs, ~] = fileparts(fov_nms);
disp("FOVs Found:"); disp(fovs)
img_info = imfinfo(fullfile(multi_dir, fov_nms(1)));
n_channels = height(struct2table(img_info));

%% Write to Single TIFs

mkdir(fullfile(single_dir));


parfor f = 1:length(fovs)
    curr_fov_nm = fovs(f);
    curr_fov_dir = fullfile(single_dir, curr_fov_nm);
    mkdir(curr_fov_dir);
    for i = 1:n_channels
        if(~channel_names.remove(i))
            curr_img = imread(fullfile(multi_dir, fov_nms(f)), i);
            imwrite(curr_img, fullfile(curr_fov_dir, strcat(channel_names.channel(i),".tif")));
        end
    end
end

%%





