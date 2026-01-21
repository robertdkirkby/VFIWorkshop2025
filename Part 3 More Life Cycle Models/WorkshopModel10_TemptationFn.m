function F=WorkshopModel10_TemptationFn(h,aprime,a,z,w,sigmatempt,scaletemptation,agej,Jr,r,kappa_j)
% Assume the temptation is just in consumption (no temptation from leisure).

F=-Inf;
% budget constraint
if agej<Jr % working
    c=(1+r)*a +w*kappa_j*h*exp(z) - aprime;
else % retired
    c=(1+r)*a -aprime;
end


if c>0
    F=scaletemptation*(c^(1-sigmatempt))/(1-sigmatempt); % The utility function
end


end
