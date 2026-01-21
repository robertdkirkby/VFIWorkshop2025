% Workshop Model 13
% Second Endogenous State: assets and housing

% Note: two endogenous states, so n_a is 1-by-2
% And action space contains (...,a1prime,a2prime,a1,a2,...)
% Divide-and-conquer and grid interpolation layer will apply to the first of the two, so is important that housing is the second endogenous state (as it has less points)

%% Model action and state-space
n_d=0; % number of grid points for our decision variable, labor supply
n_a=[201,4]; % number of grid points for our endogenous state, assets and housing
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
N_j=81; % periods, represent ages 20 to 64

%% Parameters

% Age and Retirement
Params.J=N_j; % final period
Params.agej=1:1:N_j; % model period
Params.agejshifter=19; % add to agej to get actual age
Params.Jr=65-19; % retire at age 65, which is period 46

% Preferences
Params.beta=0.98; % discount factor
Params.sigma=2; % curvature of consumption in utility fn
Params.theta=0.5;  % relative importance of consumption (vs housing services) in utility fn 
Params.upsilon=0; % substitability of consumption and housing services

% Prices
Params.r=0.05;
Params.w=1; % wage
Params.p=0.2; % rental price of housing services
% Note: implicitly, relative price of house vs assets is 1 (which is silly)

% Housing
Params.gamma=0.8; % fraction of housing that can be used as collateral
Params.phi=0.05; % housing transaction costs (as fraction of h)

% Deterministic earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% hump-shaped, then zero in retirement

% Pensions
Params.pension=0.2;

% Exogenous AR(1) on labor productivity per time worked
Params.rho_z=0.9; % autocorrelation coefficient
Params.sigma_z_epsilon=0.1; % std dev of innovations


%% Grids

d_grid=linspace(0,1,n_d)'; % note, implicitly imposes the 0<h<1 constraint

asset_grid=10*linspace(0,1,n_a(1))'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

house_grid=[0; 1; 2; 3]; % just some arbitrary house sizes

a_grid=[asset_grid; house_grid]; % stacked column vector

% Discretize AR(1) using Farmer-Toda method
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);


%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta'};

% ReturnFn
ReturnFn=@(aprime,hprime,a,h,z,w,r,p,kappa_j,pension,sigma, upsilon, theta, gamma, phi, agej, Jr)...
    WorkshopModel13_ReturnFn(aprime,hprime,a,h,z,w,r,p,kappa_j,pension,sigma, upsilon, theta, gamma, phi, agej, Jr);
% First inputs are 'action space', here (h,aprime,hprime,a,house,z,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
% Use both divide-and-conquer and grid interpolation layer. They are applied to the 'first' endogenous state.
vfoptions.divideandconquer=1;
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=30;
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% When using grid interpolation layer, have to tell simoptions too
simoptions.gridinterplayer=vfoptions.gridinterplayer;
simoptions.ngridinterp=vfoptions.ngridinterp;

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,1,ceil(n_z/2))=1; % start with 0 assets, no house, median z shock

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationary Distribution
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(aprime,hprime,a,h,z,w,kappa_j) w*kappa_j*h*exp(z); % the labor earnings
FnsToEvaluate.assets=@(aprime,hprime,a,h,z) a; % a is the current asset holdings
FnsToEvaluate.house=@(aprime,hprime,a,h,z) h; % house is the current house holdings
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Note how we use names earnings, assets, and housing

%% Calculate various stats
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);

% for example,
fprintf('Aggregate (Mean) earnings=%1.4f \n', AllStats.earnings.Mean) % Note: mass of agents is one, hence aggregate and mean are the same thing
fprintf('Gini coeff of assets=%1.2f \n', AllStats.assets.Gini)

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

figure(1);
subplot(3,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
title('Age-conditional mean of earnings')
subplot(3,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.assets.Mean)
title('Age-conditional mean of assets')
subplot(3,1,3); plot(Params.agejshifter+Params.agej, AgeConditionalStats.house.Mean)
title('Age-conditional mean of housing')


