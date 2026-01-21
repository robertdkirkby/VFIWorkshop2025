function c=WorkshopModel11_ConsumptionLagFn(h,aprime,a,z,w,agej,Jr,r,kappa_j)

% budget constraint
if agej<Jr % working
    c=(1+r)*a +w*kappa_j*h*exp(z) - aprime;
else % retired
    c=(1+r)*a -aprime;
end

end
