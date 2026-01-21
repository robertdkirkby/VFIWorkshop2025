function F=WorkshopModel11_ReturnFn(h,aprime,a,clag,z,w,sigma,eta,psi,agej,Jr,r,kappa_j,lambda,mu,upsilon,theta)

F=-Inf;

% budget constraint
if agej<Jr % working
    c=(1+r)*a +w*kappa_j*h*exp(z) - aprime;
else % retired
    c=(1+r)*a -aprime;
end

if c>0
    u_c=(c^(1-sigma))/(1-sigma); % The utility function
    u_clag=(clag^(1-sigma))/(1-sigma); % The utility function
    Delta=u_c-u_clag;
    if c>=clag % so u_c>=u_clag, Delta>=0
        v=(1-exp(-mu*Delta))/mu;
    else % c<clag, so u_c<u_clag, Delta<0
        v=-lambda*(1-exp((upsilon/lambda)*Delta))/upsilon;
    end
    F=theta*u_c+(1-theta)*v-psi*(h^(1+eta))/(1+eta);
end

% % To test, I make sure clag does nothing if using a basic setup
% if c>0
%     F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta))/(1+eta); % The utility function
% end


end
