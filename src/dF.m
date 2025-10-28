function g = dF(x,y,A,AH,F_mask)
% =========================================================================
% Gradient of the data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity image.
%           - A   : The sampling operator.
%           - AH  : Hermitian of A.
% Output:   - g   : Wirtinger gradient.
% =========================================================================
u = A(F_mask,x);
u = (abs(u) - sqrt(y)) .* exp(1i*angle(u));
g = 1/2 * AH(F_mask,u);
end