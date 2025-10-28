function v = F(x,y,A,F_mask)
% =========================================================================
% Data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - x   : The complex-valued transmittance of the sample.
%           - y   : Intensity image.
%           - A   : The sampling operator.
% Output:   - v   : Value of the fidelity function.
% =========================================================================
v = 1/2 * norm2(abs(A(F_mask,x)) - sqrt(y))^2;
end

function n = norm2(x)   % calculate the l2 vector norm
n = norm(x(:),2);
end