function gaussian_window = Create_gaussian_spectrum(center_wavelength,FWHM,wavelength_seq)
    sigma = FWHM/(2*sqrt(2*log(2)));      % 从FWHM换算标准差
    delta_lambda = wavelength_seq(2) - wavelength_seq(1); % 波长间隔 (0.5nm)
    gaussian_window = exp(-(wavelength_seq - center_wavelength).^2/(2*sigma^2));
    gaussian_window = gaussian_window/(sum(gaussian_window)*delta_lambda);
    gaussian_window=gaussian_window./sum(gaussian_window);
end

