function F=WorkshopModel14_ReturnFn(l,h,z,sigma,w)
% first four entries are the action space

F=-Inf;

% budget constraint
c=w*l*h*exp(z);
% note: trade-off between working today (l) and accumulating human-capital
% for higher future earnings (1-l)

if c>0
    % utility fn
    F=(c^(1-sigma))/(1-sigma);
end
