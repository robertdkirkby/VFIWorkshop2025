function F=WorkshopOLGModel3_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,alpha,delta,kappa_j,gamma_i,Beq,agej, Jr, Jbeq1, Jbeq2)
% first three entries are the action space

F=-Inf;

w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));

% budget constraint
if agej<Jr % working
    c=(1+r)*a +w*kappa_j*gamma_i*h*exp(z) + Beq*(agej>=Jbeq1)*(agej<=Jbeq2)- aprime;
else % retired
    c=(1+r)*a -aprime;
end

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta))/(1+eta);
end
