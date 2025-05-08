function F=WorkshopExercise1_ReturnFn(h,aprime,a,z,sigma,psi,eta,w,r,tau,kappa_j,agej, Jr)
% Note: modify line 8 to include the tax
% And modify the inputs (line 1) so that tau is an input
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
