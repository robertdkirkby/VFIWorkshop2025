function F=WorkshopExercise2_ReturnFn(h,aprime,a,e,sigma,psi,eta,w,r,kappa_j,agej, Jr)
% Modify line 8, to remove z
% Modify inputs, to remove z
F=-Inf;

% budget constraint
if agej<Jr % working
    c=(1+r)*a +w*kappa_j*h*exp(e) - aprime;
else % retired
    c=(1+r)*a -aprime;
end

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta))/(1+eta);
end
