function F=WorkshopOLGTPath1_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,w,tau,kappa_j,agej, Jr)
% first three entries are the action space

F=-Inf;

% budget constraint
if agej<Jr % working
    c=(1+r)*a +(1-tau)*w*kappa_j*h*exp(z) - aprime;
else % retired
    c=(1+r)*a -aprime;
end

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta))/(1+eta);
end
