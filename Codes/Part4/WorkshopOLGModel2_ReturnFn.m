function F=WorkshopOLGModel2_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,w,kappa_j,agej, Jr, tau_e, tau_i1, tau_i2, pension)
% first three entries are the action space

F=-Inf;

% budget constraint
if agej<Jr % working
    earnings=w*kappa_j*h*exp(z);
    income=r*a+earnings;
    incometax=(1-tau_i1*income^tau_i2)*income; % This is the tax fn of Heathcote,Storeslettern & Violante (2017)
    earningstax=tau_e*earnings;
    c=a+ income -earningstax -incometax - aprime;
else % retired
    income=r*a;
    incometax=(1-tau_i1*income^tau_i2)*income; % This is the tax fn of Heathcote,Storeslettern & Violante (2017)
    c=a+income - incometax +pension -aprime;
end

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta))/(1+eta);
end
