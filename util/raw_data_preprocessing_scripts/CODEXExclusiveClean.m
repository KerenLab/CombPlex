clear all;
% Parameters to adjust
ProjectDir = '/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Compressed sensing/Experiments/CODEX/3_BreastCA-TMA_stain2/alignment/';

dirPath = fullfile(ProjectDir,"Clean");
savePath = fullfile(ProjectDir,"For_Lior");
CompMatrix= readtable(fullfile(ProjectDir,"Matrix.csv"),TextType="string",ReadRowNames=true,VariableNamingRule='preserve');


FOVFolder = struct2table(dir(fullfile(dirPath,'*','/')));
FOVs = extractAfter(extractAfter(unique(string(FOVFolder.folder)),dirPath),filesep);


% Create GT images from single variant images (combine singles if needed)
CompArray = table2array(CompMatrix);
for j=1:length(FOVs)
    Curr_FOV = FOVs(j);
    mkdir(fullfile(savePath,Curr_FOV))
    for i=1:width(CompMatrix)
        Curr_Target = string(CompMatrix.Properties.VariableNames(i));
        Curr_Channels = string(CompMatrix.Properties.RowNames(CompArray(:,i)==1));
        if(isempty(Curr_Channels))
            error("No Channels found in column");
        elseif(length(Curr_Channels) == 1)
            imwrite(uint16(imread(fullfile(dirPath, Curr_FOV, strcat(Curr_Target, "_Clean.tif")))), fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_GT.tif")));
        else % 2 or more channels
            comb_mask = uint16(imread(fullfile(dirPath, Curr_FOV, strcat(Curr_Target, "_", Curr_Channels(1) ,"_Clean.tif"))));
            for h=2:length(Curr_Channels)
                curr_mask = uint16(imread(fullfile(dirPath, Curr_FOV, strcat(Curr_Target, "_", Curr_Channels(h) ,"_Clean.tif"))));
                comb_mask = max(cat(3, comb_mask, curr_mask), [], 3);
            end
            imwrite(comb_mask,fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_GT.tif")));
        end
    end
end

% create simulated multis from GT images according to matrix
for j=1:length(FOVs)
    Curr_FOV = FOVs(j);
    for i=1:height(CompMatrix)
        Curr_Target = string(CompMatrix.Properties.VariableNames(CompArray(i,:)==1));
        Curr_Channels = string(CompMatrix.Properties.RowNames(i));
        if (isempty(Curr_Target))
            error("No Targets found in row");
        else
            comb_target = uint16(imread(fullfile(savePath, Curr_FOV, strcat(Curr_Target(1), "_GT.tif"))));
            for h=2:length(Curr_Target)
                curr_target = uint16(imread(fullfile(savePath, Curr_FOV, strcat(Curr_Target(h),"_GT.tif"))));
                comb_target = max(cat(3, comb_target, curr_target), [], 3);
            end
            imwrite(comb_target, fullfile(savePath, Curr_FOV, strcat(Curr_Channels, "_Sim_Multi.tif")));
        end
    end
end

% % Clean real multis using simulated multis
% for j=1:length(FOVs)
%     Curr_FOV = FOVs(j);
%     for i=1:height(CompMatrix)
%         Curr_Multi = string(CompMatrix.Properties.VariableNames(CompArray(i,:)==1));
%         Curr_Channels = string(CompMatrix.Properties.RowNames(i));
%         Pos_Perms = strcat(join(perms(Curr_Multi),"-"),".tif");
%         MultiName = Pos_Perms(isfile(fullfile(dirPath,Curr_FOV,Pos_Perms)));
%         Raw_Multi = uint16(imread(fullfile(dirPath,Curr_FOV,MultiName)));
%         Sim_Multi = imread(fullfile(savePath,Curr_FOV,strcat(Curr_Channels,"_Sim_Multi.tif")));
%         imwrite(Raw_Multi .* uint16(Sim_Multi>0),fullfile(savePath, Curr_FOV, strcat(extractBefore(MultiName,".tif"),"_Clean.tif")));
%     end
% end

% Intersect GT with raw singles
for j=1:length(FOVs)
    Curr_FOV = FOVs(j);
    for i=1:width(CompMatrix)
        Curr_Target = string(CompMatrix.Properties.VariableNames(i));
        Curr_Channels = string(CompMatrix.Properties.RowNames(CompArray(:,i)==1));
        if(isempty(Curr_Channels))
            error("No Channels found in column");
        elseif(length(Curr_Channels) == 1)
            imwrite(uint16(imread(fullfile(dirPath, Curr_FOV, strcat(Curr_Target, "_Clean.tif")))), fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_union.tif")));
        else % 2 or more channels
            GT = imread(fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_GT.tif")));
            for h=1:length(Curr_Channels)
                Curr_Single = uint16(imread(fullfile(dirPath,Curr_FOV,strcat(Curr_Target,"_",Curr_Channels(h),"_raw.tif"))));
                imwrite(Curr_Single .* uint16(GT>0), fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_",Curr_Channels(h),"_union.tif")))
            end
        end
    end
end

%Intersect best clean version with raw singles
% for j=1:length(FOVs)
%     Curr_FOV = FOVs(j);
%     for i=1:width(CompMatrix)
%         Curr_Target = string(CompMatrix.Properties.VariableNames(i));
%         Curr_Channels = string(CompMatrix.Properties.RowNames(CompArray(:,i)==1));
%         if(isempty(Curr_Channels))
%             error("No Channels found in column");
%         elseif(length(Curr_Channels) == 1)
%             imwrite(uint16(imread(fullfile(dirPath, Curr_FOV, "TIFs", strcat(Curr_Target, "_clean.tif")))), fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_best.tif")));
%         else % 2 or more channels
%             if (Curr_Target == "SMA") || (Curr_Target == "p53") || (Curr_Target =="HLA-DR") 
%                 GT = imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_647_clean.tif")));
%                 for h=1:length(Curr_Channels)
%                     Curr_Single = uint16(imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_",Curr_Channels(h),".tif"))));
%                     imwrite(Curr_Single .* uint8(GT>0), fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_",Curr_Channels(h),"_best.tif")));
%                 end
%             elseif (Curr_Target == "Pan-Keratin")
%                 GT = imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_550_clean.tif")));
%                 for h=1:length(Curr_Channels)
%                     Curr_Single = uint16(imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_",Curr_Channels(h),".tif"))));
%                     imwrite(Curr_Single .* uint16(GT>0), fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_",Curr_Channels(h),"_best.tif")));
%                 end
%             end
%         end
%     end
% end

%%
% 
% ProjectDir = '/Users/benuri/Dropbox (Weizmann Institute)/Raz/Leeat Keren/Compressed sensing/Experiments/CODEX/3_BreastCA-TMA_stain2/QuPathProject/';
% 
% dirPath = fullfile(ProjectDir,"Extracted");
% savePath = fullfile(ProjectDir,"Clean");
% CompMatrix= readtable(fullfile(ProjectDir,"Matrix.csv"),TextType="string",ReadRowNames=true,VariableNamingRule='preserve');
% 
% 
% FOVFolder = struct2table(dir(fullfile(dirPath,'*','TIFs')));
% FOVs = extractBefore(extractAfter(extractAfter(unique(string(FOVFolder.folder)),dirPath),filesep),filesep);
% 
% 
% % Create GT images from single variant images (combine singles if needed)
% CompArray = table2array(CompMatrix);
% for j=1:length(FOVs)
%     Curr_FOV = FOVs(j);
%     mkdir(fullfile(savePath,Curr_FOV))
%     for i=1:width(CompMatrix)
%         Curr_Target = string(CompMatrix.Properties.VariableNames(i));
%         Curr_Channels = string(CompMatrix.Properties.RowNames(CompArray(:,i)==1));
%         if(isempty(Curr_Channels))
%             error("No Channels found in column");
%         elseif(length(Curr_Channels) == 1)
%             imwrite(uint8(imread(fullfile(dirPath, Curr_FOV, "TIFs", strcat(Curr_Target, "_clean.tif")))), fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_GT.tif")));
%         else % 2 or more channels
%             comb_mask = uint8(imread(fullfile(dirPath, Curr_FOV, "TIFs", strcat(Curr_Target, "_", Curr_Channels(1) ,"_clean.tif"))));
%             for h=2:length(Curr_Channels)
%                 curr_mask = uint8(imread(fullfile(dirPath, Curr_FOV, "TIFs", strcat(Curr_Target, "_", Curr_Channels(h) ,"_clean.tif"))));
%                 comb_mask = max(cat(3, comb_mask, curr_mask), [], 3);
%             end
%             imwrite(comb_mask,fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_GT.tif")));
%         end
%     end
% end
% 
% % create simulated multis from GT images according to matrix
% for j=1:length(FOVs)
%     Curr_FOV = FOVs(j);
%     for i=1:height(CompMatrix)
%         Curr_Target = string(CompMatrix.Properties.VariableNames(CompArray(i,:)==1));
%         Curr_Channels = string(CompMatrix.Properties.RowNames(i));
%         if (isempty(Curr_Target))
%             error("No Targets found in row");
%         else
%             comb_target = uint8(imread(fullfile(savePath, Curr_FOV, strcat(Curr_Target(1), "_GT.tif"))));
%             for h=2:length(Curr_Target)
%                 curr_target = uint8(imread(fullfile(savePath, Curr_FOV, strcat(Curr_Target(h),"_GT.tif"))));
%                 comb_target = max(cat(3, comb_target, curr_target), [], 3);
%             end
%             imwrite(comb_target, fullfile(savePath, Curr_FOV, strcat(Curr_Channels, "_Sim_Multi.tif")));
%         end
%     end
% end
% 
% % Clean real multis using simulated multis
% for j=1:length(FOVs)
%     Curr_FOV = FOVs(j);
%     for i=1:height(CompMatrix)
%         Curr_Multi = string(CompMatrix.Properties.VariableNames(CompArray(i,:)==1));
%         Curr_Channels = string(CompMatrix.Properties.RowNames(i));
%         Pos_Perms = strcat(join(perms(Curr_Multi),"-"),".tif");
%         MultiName = Pos_Perms(isfile(fullfile(dirPath,Curr_FOV,"TIFs",Pos_Perms)));
%         Raw_Multi = uint8(imread(fullfile(dirPath,Curr_FOV,"TIFs",MultiName)));
%         Sim_Multi = imread(fullfile(savePath,Curr_FOV,strcat(Curr_Channels,"_Sim_Multi.tif")));
%         imwrite(Raw_Multi .* uint8(Sim_Multi>0),fullfile(savePath, Curr_FOV, strcat(extractBefore(MultiName,".tif"),"_Clean.tif")));
%     end
% end
% 
% % Intersect GT with raw singles
% for j=1:length(FOVs)
%     Curr_FOV = FOVs(j);
%     for i=1:width(CompMatrix)
%         Curr_Target = string(CompMatrix.Properties.VariableNames(i));
%         Curr_Channels = string(CompMatrix.Properties.RowNames(CompArray(:,i)==1));
%         if(isempty(Curr_Channels))
%             error("No Channels found in column");
%         elseif(length(Curr_Channels) == 1)
%             imwrite(uint8(imread(fullfile(dirPath, Curr_FOV, "TIFs", strcat(Curr_Target, "_clean.tif")))), fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_union.tif")));
%         else % 2 or more channels
%             GT = imread(fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_GT.tif")));
%             for h=1:length(Curr_Channels)
%                 Curr_Single = uint8(imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_",Curr_Channels(h),".tif"))));
%                 imwrite(Curr_Single .* uint8(GT>0), fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_",Curr_Channels(h),"_union.tif")))
%             end
%         end
%     end
% end
% 
% %Intersect best clean version with raw singles
% for j=1:length(FOVs)
%     Curr_FOV = FOVs(j);
%     for i=1:width(CompMatrix)
%         Curr_Target = string(CompMatrix.Properties.VariableNames(i));
%         Curr_Channels = string(CompMatrix.Properties.RowNames(CompArray(:,i)==1));
%         if(isempty(Curr_Channels))
%             error("No Channels found in column");
%         elseif(length(Curr_Channels) == 1)
%             imwrite(uint8(imread(fullfile(dirPath, Curr_FOV, "TIFs", strcat(Curr_Target, "_clean.tif")))), fullfile(savePath, Curr_FOV, strcat(Curr_Target, "_best.tif")));
%         else % 2 or more channels
%             if (Curr_Target == "SMA") || (Curr_Target == "p53") || (Curr_Target =="HLA-DR") 
%                 GT = imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_647_clean.tif")));
%                 for h=1:length(Curr_Channels)
%                     Curr_Single = uint8(imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_",Curr_Channels(h),".tif"))));
%                     imwrite(Curr_Single .* uint8(GT>0), fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_",Curr_Channels(h),"_best.tif")));
%                 end
%             elseif (Curr_Target == "Pan-Keratin")
%                 GT = imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_550_clean.tif")));
%                 for h=1:length(Curr_Channels)
%                     Curr_Single = uint8(imread(fullfile(dirPath,Curr_FOV,"TIFs",strcat(Curr_Target,"_",Curr_Channels(h),".tif"))));
%                     imwrite(Curr_Single .* uint8(GT>0), fullfile(savePath,Curr_FOV,strcat(Curr_Target,"_",Curr_Channels(h),"_best.tif")));
%                 end
%             end
%         end
%     end
% end
% 
% 
% 
% 
% 
