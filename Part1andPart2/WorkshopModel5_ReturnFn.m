function F=WorkshopModel5_ReturnFn(h,aprime,a,z,sigma,psi,eta,w,r,kappa_j,alpha_i,agej, Jr)
% first four entries are the action space

F=-Inf;

% budget constraint
if agej<Jr % working
    c=(1+r)*a +w*kappa_j*h*exp(alpha_i+z) - aprime;
else % retired
    c=(1+r)*a -aprime;
end

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta))/(1+eta);
end
