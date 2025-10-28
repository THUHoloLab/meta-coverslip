function [test_img_num, object] = Load_test_img_Holo(N, i)
    % 定义基础文件夹路径
    base_dir = 'E:/Hololab/researches/Nonlocal_Multimodal/Rec_algorithm/test_img_dataset';
    amplitude_dir = fullfile(base_dir, 'gt_amplitude');
    phase_dir = fullfile(base_dir, 'gt_phase');
    
    % 获取并排序两个文件夹中的 .mat 文件
    amplitude_files = dir(fullfile(amplitude_dir, '*.mat'));
    amplitude_files = sort({amplitude_files.name});
    phase_files = dir(fullfile(phase_dir, '*.mat'));
    phase_files = sort({phase_files.name});
    
    % 计算测试图片总数
    test_img_num = length(amplitude_files);
    
    % 检查索引 i 是否有效
    if i < 1 || i > test_img_num
        error('索引 i 超出范围，必须在 1 到 %d 之间', test_img_num);
    end
    
    % 构造第 i 张图片的文件路径
    amplitude_file = fullfile(amplitude_dir, amplitude_files{i});
    phase_file = fullfile(phase_dir, phase_files{i});
    
    % 加载 .mat 文件
    amplitude_data = load(amplitude_file);
    phase_data = load(phase_file);
    
    % 提取 .mat 文件中的变量（假设每个文件中只有一个变量）
    amplitude_var = fieldnames(amplitude_data);
    amplitude = amplitude_data.(amplitude_var{1});
    phase_var = fieldnames(phase_data);
    phase = phase_data.(phase_var{1});
    
    % 检查振幅和相位矩阵大小是否一致
    if ~isequal(size(amplitude), size(phase))
        error('振幅和相位矩阵的大小不一致');
    end
    
    % 计算复振幅图像
    object = amplitude .* exp(1i * phase);
    
    % 调整图像大小为 N x N
    object = imresize(object, [N, N]);
end