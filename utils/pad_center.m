% function output = pad_center(input, output_size)
%     input_size = size(input);
%     output = gpuArray(zeros(output_size));
%     output(...
%     ceil(end/2 - input_size(1)/2 + 1:end/2 + input_size(1)/2),...
%     ceil(end/2 - input_size(2)/2 + 1:end/2 + input_size(2)/2)) = input;
% end


function output = pad_center(input, output_size, method)
    % 参数说明:
    %   input:       输入矩阵（支持CPU/GPU数组）
    %   output_size: 目标输出尺寸（如[512, 512]）
    %   method:      填充方式，可选'replicate'（默认）/'symmetric'/'circular'
    
    if nargin < 3
        method = 'replicate'; % 设置默认填充方式
    end
    
    input_size = size(input);
    ndims_input = numel(input_size);
    output = input;
    
    % 遍历每个维度进行填充
    for dim = 1:ndims_input
        current_dim_size = size(output, dim);
        target_dim_size = output_size(dim);
        
        % 计算需要填充的总量
        pad_total = target_dim_size - current_dim_size;
        if pad_total <= 0
            continue; % 当前维度不需要填充
        end
        
        % 计算前后填充量
        pad_pre = floor(pad_total/2);
        pad_post = ceil(pad_total/2);
        
        % 构建填充向量（其他维度填0）
        pad_vec = zeros(1, ndims_input);
        pad_vec(dim) = pad_pre;
        
        % 前向填充
        output = padarray(output, pad_vec, method, 'pre');
        
        % 后向填充
        pad_vec(dim) = pad_post;
        output = padarray(output, pad_vec, method, 'post');
    end
    
    % 最终尺寸校准（防止计算误差）
    crop_indices = arrayfun(@(x) 1:x, output_size, 'UniformOutput', false);
    output = output(crop_indices{:});
end