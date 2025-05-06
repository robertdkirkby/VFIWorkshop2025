% Workshop Model 11
% Loss Aversion (Prospect Theory)

% Households like gains (as usual), but they dislike losses even more than
% they dislike gains (which is not true in standard preferences). We need
% to define a 'reference point' that determines what are gains (above the
% reference point) and what are losses (below the reference point).
% Here we will use the 'lag of consumption' as the reference point. So the
% first change to codes is to include the 'lag of consumption' as a
% 'residualasset' (a form of endogenous state).
% The second change is to the utility fn (so ReturnFn in code). To capture
% that we dislike losses more than we like gains, the idea is to use a
% utility fn where the derivatives (marginal utility) is steeper for losses
% than for gains (so there is a discontinuity for utility as we cross the
% reference point).
% So there will be essentially two changes to the code to implement loss
% aversion. First will be setting up a residual asset (change n_a, a_grid,
% and set vfoptions.residualasset=1 and ) and the second is to the ReturnFn
% to implement the loss aversion with respect to the reference point.

% Note: Because of the residualasset this requires a more powerful GPU than most of our other examples.

%% Model action and state-space
n_d=11; % number of grid points for our decision variable, labor supply
n_a=[101,51]; % Endogenous asset holdings, lag of consumption (residual endogenous state)
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
N_j=81; % periods, represent ages 20 to 100

%% Parameters for loss aversion
Params.lambda=2.25; % the (inverse of) the importance of loss aversion
Params.mu=1; % controls how quickly the sensitivity of gains decreases at the margin with larger gains
Params.upsilon=1; % controls how quickly the sensitivity of losses decreases at the margin with larger losses
Params.theta=0.5; % indexes the degree of loss aversion, effectively determining the importance of losses relative to gains
% There is also sigma which is the CES utility coefficient
% I use the following notation
% (1-theta)*u(c_t)+theta*v(u(c_t)-u(c_t-1))
% where v(Delta)=(1-e^(-mu*Delta))/mu for Delta>=0
%       v(Delta)=-lambda*(1-e^((upsilon/lambda)*Delta))/upsilon for Delta<0
% I set
% u(c)=(c^(1-sigma))/(1-sigma)

vfoptions.residualasset=1; % Using an residual asset
simoptions.residualasset=1;

%% Parameters

% Age and Retirement
Params.J=N_j; % final period
Params.agej=1:1:N_j; % model period
Params.agejshifter=19; % add to agej to get actual age
Params.Jr=65-19; % retire at age 65, which is period 46

% Preferences
Params.beta=0.98; % discount factor
Params.sigma=2; % curvature of consumption in utility fn
Params.psi=3; % relative importance of labor supply in utility fn
Params.eta=0.5;  % curvature of labor supply in utility fn 

% Prices
Params.r=0.05;
Params.w=1; % wage

% Deterministic earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% hump-shaped, then zero in retirement

% Exogenous AR(1) on labor productivity per time worked
Params.rho_z=0.9; % autocorrelation coefficient
Params.sigma_z_epsilon=0.1; % std dev of innovations

% Conditional survival probabilities
Params.sj=[ones(1,46),linspace(1,0.99,81-46-10),linspace(0.99,0.9,9),0];

%% Grids

d_grid=linspace(0,1,n_d)'; % note, implicitly imposes the 0<h<1 constraint

asset_grid=10*linspace(0,1,n_a(1))'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

clag_grid=5*(linspace(0,1,n_a(2)).^3)';

% Discretize AR(1) using Farmer-Toda method
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);


a_grid=[asset_grid; clag_grid];


%% Set up the residual asset, we need to expain how to get value of rprime given (d,aprime,a,z) (this model does not have d)
% Note that the residual asset is NOT one of the inputs
vfoptions.rprimeFn=@(h,aprime,a,z,w,agej,Jr,r,kappa_j) WorkshopModel11_ConsumptionLagFn(h,aprime,a,z,w,agej,Jr,r,kappa_j); % consumption (today, as this is then next period's lag of consumption)
simoptions.rprimeFn=vfoptions.rprimeFn;
% Because we need all the grids to evaluate rprimeFn these have to be passed via simoptions
simoptions.d_grid=d_grid;
simoptions.a_grid=a_grid;
simoptions.z_grid=z_grid;


%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta','sj'};

% ReturnFn
ReturnFn=@(h,aprime,a,clag,z,w,sigma,eta,psi,agej,Jr,r,kappa_j,lambda,mu,upsilon,theta)...
    WorkshopModel11_ReturnFn(h,aprime,a,clag,z,w,sigma,eta,psi,agej,Jr,r,kappa_j,lambda,mu,upsilon,theta);
% First inputs are 'action space', here (h,aprime,a,z,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,1,ceil(n_z/2))=1; % start with 0 assets, median z shock
% Note: Not obvious what you should set for reference point in first period

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationart Distribution
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(h,aprime,a,clag,z,w,kappa_j) w*kappa_j*h*exp(z); % the labor earnings
FnsToEvaluate.assets=@(h,aprime,a,clag,z) a; % a is the current asset holdings
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Note how we use names earnings and assets

%% Calculate various stats
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);

% for example,
fprintf('Aggregate (Mean) earnings=%1.4f \n', AllStats.earnings.Mean) % Note: mass of agents is one, hence aggregate and mean are the same thing
fprintf('Gini coeff of assets=%1.2f \n', AllStats.assets.Gini)

% Note how it uses the names we gave the FnsToEvaluate

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);


figure(1);
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
title('Age-conditional mean of earnings')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.assets.Gini)
title('Age-conditional Gini coeff of assets')
% Doing much the same as AllStats, except now it is conditional on age
% Note: now people differ by age


