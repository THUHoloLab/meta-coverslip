function im=Load_initial_image(N)
% %     Load test image
% 
 %  Coins
    phase = double(imread("eight.tif"));
    phase=phase(:,1:size(phase,1));
    amplitude = double(imread("eight.tif"));
    amplitude=amplitude(:,1:size(amplitude,1));
    phase=255-phase;

    phase=phase./255;
    phase=(phase)*2;
    amplitude=(amplitude./255)/2+0.4;
%  %%%%%%%  USAF phase target
%     phase=double(imread("USAF-1951.png"));
%     phase=phase./(max(max(phase)));
%     phase = phase - min(min(phase));
%     phase=-phase;
%     phase=phase-phase(1,1);
%     amplitude=ones(size(phase));


% %%%%%%%  USAF amplitude target   
%     amplitude = double(imread("USAF-1951.png"));
%     
%     amplitude=amplitude./(max(max(amplitude)));
%     amplitude=~logical(amplitude);
%     phase=zeros(size(amplitude));
%%  Cameraman
%     phase=double(imread("cameraman.tif"));
%     amplitude = double(imread("cameraman.tif"));
% %     
% 
% %%     catoon_cell
%     phase=double(rgb2gray(imread("cell_cartoon.png")));
%     amplitude = 3000-double(rgb2gray(imread("cell_cartoon.png")));

    
%%  Random surface
%    phase=generateSmoothRandomSurface(N);

 %%   
    % Resize images
    phase = imresize(phase, [N, N]);
    amplitude = imresize(amplitude, [N, N]);
    im = amplitude .* exp(1i .*(pi/2.* phase));
end


function Z = generateSmoothRandomSurface(N)
    % 设置随机种子
    rng('shuffle');

    % 生成大小为 N x N 的随机高斯噪声
    noise = randn(N, N);

    % 对随机噪声进行二维傅里叶变换
    F_noise = fft2(noise);

    % 生成高斯低通滤波器
    [X, Y] = meshgrid(1:N, 1:N);
    X = X - ceil(N / 2);
    Y = Y - ceil(N / 2);
    sigma = N / 500;  % 控制平滑程度
    G = exp(-(X.^2 + Y.^2) / (2 * sigma^2));

    % 应用滤波器
    F_filtered = F_noise .* fftshift(G);

    % 反傅里叶变换回到空间域
    Z = real(ifft2(F_filtered));

    % 归一化高度到 [0, 1] 范围
    Z = Z - min(Z(:));
    Z = Z / max(Z(:));

    % 高度最高为1
    Z = Z;
end
