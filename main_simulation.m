clear;clc
addpath('colorplus\')
addpath('utils\')
addpath("src\")
%% Parameters
N = 256;
dmin = 5500/20; %nm
dmax = N * dmin;
fmax = 1 / dmin;
fmin = fmax / N;

x=-dmax/2:dmin:dmax/2-dmin;y=-dmax/2:dmin:dmax/2-dmin;
fx=-fmax/2:fmin:fmax/2-fmin;fy=-fmax/2:fmin:fmax/2-fmin;
[xm, ym] = meshgrid(x,y);
[fxm, fym] = meshgrid(fx,fy);

%% Define functions

ft = @(x) fftshift(fft2(ifftshift(x)));
ift = @(x) fftshift(ifft2(ifftshift(x)));
Crop= @(x) crop_center(x,[N,N]);
Pad= @(x) pad_center(x,[2*N,2*N]);
% 
A= @(F_mask,obj) (ift(ft(obj).*F_mask));   %  input obj is transmission 
AH= @(F_mask,obj) (ift(ft(obj).*conj(F_mask)));  %  input obj is transmission 

%% Load dispersion
k_over_k0=0:0.001:0.42;
x_angles_seq = asin(k_over_k0).*180/pi;
wavelength_seq = 400:0.5:700;

%  Transmisison average polar Experiment
Ts_data=load('E:\Hololab\researches\Nonlocal_Multimodal\Dispersion_measurement_v4thinfilm\500um_Dispersion_s_polar.mat',"x_angle_seq","wavelength_seq","Ts");
Tp_data=load('E:\Hololab\researches\Nonlocal_Multimodal\Dispersion_measurement_v4thinfilm\500um_Dispersion_p_polar.mat',"x_angle_seq","wavelength_seq","Tp");
Ts=Ts_data.Ts;Tp=Tp_data.Tp;
wavelength=Ts_data.wavelength_seq;x_angles=Ts_data.x_angle_seq;
% 
[wavelength_grid,x_angles_grid]=ndgrid(wavelength, x_angles);
[wavelength_seq_grid,x_angles_seq_grid]=ndgrid(wavelength_seq, x_angles_seq);

Tp_interp = interpn(wavelength_grid, x_angles_grid, Tp, wavelength_seq_grid, x_angles_seq_grid, 'spline');
Ts_interp = interpn(wavelength_grid, x_angles_grid, Ts, wavelength_seq_grid, x_angles_seq_grid, 'spline');
T=(Tp_interp+Ts_interp)./2;
% Draw_single_profile(k_over_k0,wavelength_seq,T,'Transmisison average polar Experiment','amplitude','kx/k0','wavelength(nm)'  )


% %  Transmisison average polar Simulation
% load('E:\Hololab\researches\Nonlocal_Multimodal\Thinfilm_Design\Thinfilm design_multifunctional_v4\T_simu_v4.mat','T_s','T_p','wavelength','x_angles')
% [wavelength_grid,x_angles_grid]=ndgrid(wavelength, x_angles);
% [wavelength_seq_grid,x_angles_seq_grid]=ndgrid(wavelength_seq, x_angles_seq);
% omega_k_Tp=T_p;omega_k_Ts=T_s;
% 
% Tp_interp = interpn(wavelength_grid, x_angles_grid, omega_k_Tp, wavelength_seq_grid, x_angles_seq_grid, 'spline');
% Ts_interp = interpn(wavelength_grid, x_angles_grid, omega_k_Ts, wavelength_seq_grid, x_angles_seq_grid, 'spline');
% T=(Tp_interp+Ts_interp)./2;
% Draw_single_profile(k_over_k0,wavelength_seq,T,'Transmisison average polar Simulation','amplitude','kx/k0','wavelength(nm)'  )


%  Phase average polar Simulation
wavelength=wavelength*1000;
load('E:\Hololab\researches\Nonlocal_Multimodal\Thinfilm_Design\Thinfilm design_multifunctional_v4\phi_simu_v4.mat','phi_s','phi_p','wavelength','x_angles')
[wavelength_grid,x_angles_grid]=ndgrid(wavelength, x_angles);
[wavelength_seq_grid,x_angles_seq_grid]=ndgrid(wavelength_seq, x_angles_seq);

phi_s=phi_s;phi_p=phi_p;
phi_p_interp = interpn(wavelength_grid, x_angles_grid, phi_p, wavelength_seq_grid, x_angles_seq_grid, 'spline');
phi_s_interp = interpn(wavelength_grid, x_angles_grid, phi_s, wavelength_seq_grid, x_angles_seq_grid, 'spline');
phi=(phi_p_interp+phi_s_interp)./2;

phi=phi-phi(:,1)*ones(size(phi(1,:)));
phi=-phi;
% Draw_single_profile(k_over_k0,wavelength_seq,phi,'Phase average polar Simulation','phase','kx/k0','wavelength(nm)'  )


% FSP correction
z=100000;
[wavelength_seq_grid,k_over_k0_grid]=ndgrid(wavelength_seq, k_over_k0);
omega_k_phi_FSP=z*2*pi./wavelength_seq_grid.*sqrt(1-(k_over_k0_grid).^2);
omega_k_phi_FSP=omega_k_phi_FSP-omega_k_phi_FSP(:,1).*ones(1,size(omega_k_phi_FSP,2));


phi=omega_k_phi_FSP+phi;
%Draw_single_profile(k_over_k0,wavelength_seq,omega_k_phi,'omega k phi FSP compensation','amplitude','kx/k0','wavelength(nm)'  )

%%


noise_level_seq=[0,1e-4,5e-4,1e-3,5e-3,1e-2,2.5e-2,5e-2,0.1,0.25,0.5];
rec_config_modes={'Double_Pass','High_Pass','Low_Pass','All_Pass'};
% 
% noise_level_seq=[1e-4];
% rec_config_modes={'Double_Pass'};

%% Load_initial_image and simulation
[test_img_num,~]=Load_test_img_DIV2K(N,1);


for rec_config = rec_config_modes

if strcmp(rec_config,'Double_Pass')
    opt.center_wavelengths = [525,515]; % nm
    opt.FWHM = 10; % nm
    num_measurements = length(opt.center_wavelengths);
elseif strcmp(rec_config,'High_Pass')
    opt.center_wavelengths = [515]; % nm
    opt.FWHM = 10; % nm
    num_measurements = length(opt.center_wavelengths);
elseif strcmp(rec_config,'Low_Pass')
    opt.center_wavelengths = [525]; % nm
    opt.FWHM = 10; % nm
    num_measurements = length(opt.center_wavelengths);
else
    num_measurements=1;
end



for noise_level = noise_level_seq
opt.noise_level=noise_level;

for test_img_idx=1:test_img_num
[test_img_num,object]=Load_test_img_DIV2K(N,test_img_idx);

% AP_Draw(x,y,abs(object),angle(object),'abs','phase','','')

k_response_amplitude=cell(1, num_measurements);
k_response_phase=cell(1, num_measurements);
s=cell(1, num_measurements);
measurements = cell(1, num_measurements);
for i = 1:num_measurements

    if ~strcmp(rec_config,'All_Pass')
        [k_response_amplitude{i}, k_response_phase{i},fourier_masks{i}] = Calculate_F_mask(opt.center_wavelengths(i), opt.FWHM, wavelength_seq, T, phi, k_over_k0, fxm, fym);
    end
    

    if strcmp(rec_config,'All_Pass')
        [k_response_amplitude{i}, k_response_phase{i},fourier_masks{i}] = Calculate_F_mask(520, 10, wavelength_seq, T, phi, k_over_k0, fxm, fym);
        fourier_masks{i}=logical(abs(fourier_masks{i})).*exp(1i.*angle(fourier_masks{i}));
    end
    
        measurements{i}=abs(A(fourier_masks{i},object)).^2;
        measurements{i}=measurements{i}+rand(N).*max(measurements{i}(:))*opt.noise_level;
end


%% ——— Move to GPU———
gpuDevice(1);
x_obj        = gpuArray( AH(fourier_masks{1}, measurements{1}) );

% x_obj        = ones(size(measurements{2})).*exp(1i.*rand(size(measurements{2}))) ;
supportMask  = gpuArray( ones(size(x_obj), 'like', x_obj) );
supportMask([1:20,N-20:N],:)=0;supportMask(:,[1:20,N-20:N])=0;
for i = 1:num_measurements
    measurements{i}   = gpuArray(measurements{i});
    fourier_masks{i}  = gpuArray(fourier_masks{i});
end

%% ——— 超参数 ———
opt.num_iters = 2000;
opt.lambda_TV         = 1e-3;
opt.lambda_absorption = 1e2;
opt.lambda_support    = 0e2;
opt.descent_method = 'GD_Nesterov'; 

% GD 参数
if strcmp(opt.descent_method, 'GD') || strcmp(opt.descent_method, 'GD_Nesterov')
    opt.alpha = 1e-2;  % GD 初始学习率
    opt.momentum = 0.9;  % 动量参数
    v = gpuArray(zeros(size(x_obj)));  % 动量项
end

% Adam 参数
if strcmp(opt.descent_method, 'Adam')
    opt.alpha  = 1e-4;     % 初始学习率
    opt.beta1  = 0.9;
    opt.beta2  = 0.999;
    opt.epsilon= 1e-8;
    m = gpuArray(zeros(size(x_obj)));    % 一阶矩 (Adam 用)
    v = gpuArray(zeros(size(x_obj)));    % 二阶矩 (Adam 用)
end

%% ——— 主迭代 ———
F_loss_seq = zeros(1, opt.num_iters);
R_loss_seq = zeros(1, opt.num_iters);
MSE_seq = zeros(1, opt.num_iters);
for t = 1:opt.num_iters
    if strcmp(opt.descent_method, 'GD') || strcmp(opt.descent_method, 'Adam')

        grad_F = Grad_Fidelity_loss(x_obj, measurements, fourier_masks, A, AH);
        xDL       = dlarray(x_obj, 'SS');         % SS = spatial–spatial
        supportDL = dlarray(supportMask, 'SS');
        [regLoss, gradRegDL] = dlfeval( ...
            @computeRegularGrad, ...
             xDL, supportDL, ...
             opt.lambda_TV, opt.lambda_absorption, opt.lambda_support );
        grad_Reg = extractdata(gradRegDL);  % 已经是 gpuArray
        % 合并梯度
        grad_loss = grad_F + grad_Reg;
    
        % 根据下降方法更新 u_obj
        if strcmp(opt.descent_method, 'Adam')
            m = opt.beta1 * m + (1 - opt.beta1) * grad_loss;
            v = opt.beta2 * v + (1 - opt.beta2) * (grad_loss.^2);
            mHat = m / (1 - opt.beta1^t);
            vHat = v / (1 - opt.beta2^t);
            x_obj = x_obj - opt.alpha * mHat ./ (sqrt(vHat) + opt.epsilon);
        elseif strcmp(opt.descent_method, 'GD')
            x_obj = x_obj - opt.alpha * grad_loss;
        end
  
    elseif strcmp(opt.descent_method, 'GD_Nesterov')

        % Nesterov: 计算临时位置
        x_obj_temp = x_obj - opt.momentum * v;
        grad_F_temp = Grad_Fidelity_loss(x_obj_temp, measurements, fourier_masks, A, AH);
        
        % 自动微分计算正则化梯度（在临时位置）
        xDL_temp = dlarray(x_obj_temp, 'SS');
        supportDL = dlarray(supportMask, 'SS');
        [regLoss, gradRegDL_temp] = dlfeval(@computeRegularGrad, xDL_temp, supportDL, opt.lambda_TV, opt.lambda_absorption, opt.lambda_support);
        grad_Reg_temp = extractdata(gradRegDL_temp);
        % 合并梯度（在临时位置）
        grad_loss_temp = grad_F_temp + grad_Reg_temp;
        % 更新动量项
        v = opt.momentum * v + opt.alpha * grad_loss_temp;
        % 更新参数
        x_obj = x_obj - v;
    end
    % 记录损失
    F_loss_seq(t) = Fidelity_loss(x_obj, measurements, fourier_masks, A);
    R_loss_seq(t) = gather(extractdata(regLoss));
    MSE_seq(t) = immse(double(gather(x_obj)),object);

        %%%每 10 步打印并绘图（绘图前要 gather 回 CPU）
%     if mod(t,10)==0
%         rec_amp   = gather(abs(x_obj));
%         rec_phase = gather(angle(x_obj));
% 
%         f_a=subplot(1,3,1);
%         figure(1);
%         imagesc(rec_amp);
%         ax = gca;
%         ax.TickDir = 'in';
%         ax.YDir="normal";
%         set(gca,'FontName','Arial','FontSize',8,'LineWidth',1);
%         colormap(f_a,flip(addcolorplus(272),1));
%         c = colorbar;
%         c.Box="on";
%         c.FontSize = 8;
%         c.LineWidth=1;
%         title('Rec Amplitude')
%         
%         f_p=subplot(1,3,2);
%         imagesc(rec_phase);
%         ax = gca;
%         ax.TickDir = 'in';     ax.YDir="normal";
% %         ax.CLim=[-0.5,0.5];
%         set(gca,'FontName','Arial','FontSize',8,'LineWidth',1);
%         colormap(f_p,addcolorplus(302));
%         c = colorbar;
%         c.Box="on";
%         c.FontSize = 8;
%         c.LineWidth=1;
%         title('Rec Phase')
%         subplot(1,3,3)
%         hold on
%         plot(F_loss_seq(1:t), 'r');  % 红色曲线
%         plot(R_loss_seq(1:t), 'b');   % 蓝色曲线
%         plot(F_loss_seq(1:t)+R_loss_seq(1:t), 'k');
%         hold off
%         set(gca, 'YScale', 'log')  % 确保y轴为对数坐标
%         legend('F loss', 'R loss', 'Total loss','Location', 'best')  % 正确图例写法
%         xlabel('Iteration')          % 添加x轴标签
%         ylabel('Loss')               % 添加y轴标签
%         grid on                      % 添加网格线
%         
%         drawnow
%     end
end
gt = object;
rec = double(gather(x_obj));




save_dir = fullfile('results_TV=1e-3', ['noise_level=' num2str(opt.noise_level) 'RecMode=' rec_config{:}  ]);

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

save( ...
    fullfile(save_dir, ['img' num2str(test_img_idx)]), ...
    'rec', 'gt', 'F_loss_seq', 'R_loss_seq', 'MSE_seq', 'opt' ...
);


end
end
end

%% ——— computeRegularGrad ———
function [loss, grad] = computeRegularGrad(x_complex, supportMask, ...
                                          lambda_TV, lambda_abs, lambda_sup)
    %——— 拆分为实部和虚部 ———
    xR = real(x_complex);
    xI = imag(x_complex);

    %——— 计算 loss（都用 xR + 1i*xI 重建成复数） ———
    x = xR + 1i*xI;
    % Total Variation
    loss_TV = sum(abs(D(x)), 'all');
    % absorption 惩罚
    loss_abs = sum( relu(abs(x)-1).^2 + relu(-abs(x)).^2 , 'all');
    % support 惩罚
    loss_sup = sum( ((1 - x) .* (1 - supportMask)).^2 , 'all');

    % 加权总 loss （这是个标量实数）
    loss = real(lambda_TV*loss_TV + lambda_abs*loss_abs + lambda_sup*loss_sup);

    %——— 分别对 xR, xI 求梯度 ———
    [dLdxR, dLdxI] = dlgradient(loss, xR, xI);

    %——— 合成复数梯度 —— dLoss/dxR + i·dLoss/dxI
    grad = dLdxR + 1i * dLdxI;
end

function loss = Fidelity_loss(x,measurements,fourier_masks,A)
    loss=0;
    mea_num=length(measurements);
    for i=1:mea_num
        loss=loss+F(x,measurements{i},A,fourier_masks{i});
    end
end

function grad_loss = Grad_Fidelity_loss(x,measurements,fourier_masks,A,AH)
    grad_loss=zeros(size(measurements{1}));
    mea_num=length(measurements);
    for i=1:mea_num
        grad_loss=grad_loss+dF(x,measurements{i},A,AH,fourier_masks{i});
    end
end

function [k_response_amplitude,k_response_phase,fourier_mask] = Calculate_F_mask(center_wavelength, FWHM, wavelength_seq, T, phi, k_over_k0, fxm, fym)
    % 计算 k0
    k0 = 2 * pi / center_wavelength;
    
    % 创建高斯光谱
    illumination_spectrum = Create_gaussian_spectrum(center_wavelength, FWHM, wavelength_seq);
    
    % 计算幅度和相位响应
    k_response_amplitude = sum(T .* illumination_spectrum', 1);
    k_response_phase = sum(phi .* illumination_spectrum', 1);

    % 插值计算傅里叶掩模的幅度和相位
    fourier_mask_amplitude = interp1(k_over_k0 * k0, k_response_amplitude, 2 * pi * sqrt(fxm.^2 + fym.^2), 'linear', 0);
    fourier_mask_phase = interp1(k_over_k0 * k0, k_response_phase, 2 * pi * sqrt(fxm.^2 + fym.^2), 'linear', 0);
    
    % 计算傅里叶掩模
    fourier_mask = fourier_mask_amplitude .* exp(1i * fourier_mask_phase);
end


