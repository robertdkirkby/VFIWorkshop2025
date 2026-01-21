function F=WorkshopOLGModel4_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,w,kappa_j,agej, Jr)
% Only difference from WorkshopOLGModel1_ReturnFn is that w is taken as an input

F=-Inf;

% budget constraint
if agej<Jr % working
    c=(1+r)*a +w*kappa_j*h*exp(z) - aprime;
else % retired
    c=(1+r)*a -aprime;
end

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta))/(1+eta);
end
