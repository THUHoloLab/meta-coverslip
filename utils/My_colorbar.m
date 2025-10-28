function my_colorbar = My_colorbar(basecolor, k)
    % 输入参数:
    %   basecolor: 基础颜色（RGB三元组，范围[0,1]）
    %   k: 控制Sigmoid斜率的参数（默认值为1）
    % 输出:
    %   my_colorbar: 生成的颜色条（n×3矩阵，每行代表一个RGB颜色）


    
    n = 64; % 颜色条的分辨率
    x = linspace(0, 1, n)'; % 生成线性空间的位置值
    y=x.^0.8;

    if nargin < 2
        k = 1; % 默认斜率
    end
%     % 定义Sigmoid函数
%     sigma = @(z) 1 ./ (1 + exp(-z)); 
%     % 非线性映射并归一化到[0,1]
%     y = (sigma(k*(x - 0.5)) - sigma(-0.5*k)) / (sigma(0.5*k) - sigma(-0.5*k));
%     % 确保边界严格为0和1
%     y(x <= 0) = 0;
%     y(x >= 1) = 1;
    
    % 混合参数设定
    threshold = 0;   % 开始混合白色的阈值
    gamma = 3;         % 控制混合曲线陡峭度（越大越突然）
    max_blend = 0.95;   % 最大混合比例（防止全白）
    
    % 计算混合比例
    blend = max(0, (y - threshold)/(1 - threshold)); % 线性混合比例
    blend = blend.^gamma;        % 应用非线性调整
    blend = blend * max_blend;   % 限制最大混合比例
    
    % 混合基础颜色与白色
    my_colorbar = (1 - blend) .* y*basecolor + blend .* ones(64,3);
    
    % 确保颜色值在合法范围内
    my_colorbar = min(max(my_colorbar, 0), 1);
end
