function [combination_label_img,comb_names] = multi2CombinationLabelImg(Curr_Multi, multi_path)
    % get table of combinations
    n_multi = length(Curr_Multi);
    n_comb = 2^n_multi;
    comb_array = zeros(n_comb-1, n_multi);
    for i = 1:(n_comb-1)
        comb_array(i, :) = dec2bin(i, n_multi) - '0';
    end
    comb_table = array2table(comb_array, "VariableNames",Curr_Multi);
   
    Curr_Single = double(imbinarize(imread(fullfile(multi_path,strcat(Curr_Multi(1),"_Clean.tif")))));
    All_Singles = repmat(Curr_Single, 1, 1, n_multi);
    for i = 2:n_multi
        All_Singles(:, :, i) = double(imbinarize(imread(fullfile(multi_path,strcat(Curr_Multi(i),"_Clean.tif")))));
    end

    All_Singles_vec = reshape(All_Singles, [], n_multi);
    [~, combination_label_img] = ismember(All_Singles_vec, comb_array, "rows");
    combination_label_img = reshape(combination_label_img, size(Curr_Single));
    for i = 1:height(comb_table)
        comb_names(i) = join(string(comb_table.Properties.VariableNames(comb_array(i,:)==1)),"-");
    end

end