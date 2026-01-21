function F=WorkshopModel16_ReturnFn(savings,a,z,sigma,w,kappa_j,agej, Jr)
% first three entries are the action space

F=-Inf;

% budget constraint
if agej<Jr % working
    c=a +w*kappa_j*exp(z) - savings;
else % retired
    c=a -savings;
end

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma);
end
