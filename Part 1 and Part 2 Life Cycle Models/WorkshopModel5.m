% Workshop Model 5
% Permanent types by number, N_i.

% Eight core steps:
% 1. Model action and state-spaces.
% 2. Parameters
% 3. Grids
% 4. Permanent type masses
% 5. ReturnFn
% 6. Solve for value fn and policy fn.
% 7. Agents stationary distribution.
% 8. Generate model moments/statistics.


%% Model action and state-space
n_d=21; % number of grid points for our decision variable, labor supply
n_a=201; % number of grid points for our endogenous state, assets
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
N_j=81; % periods, represent ages 20 to 100

N_i=5; % number of permanent types

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

% Fixed-effect, alpha_i (is normally distributed across permanent types)
Params.sigma_alpha=0.5;

%% Grids

d_grid=linspace(0,1,n_d)'; % note, implicitly imposes the 0<h<1 constraint

a_grid=10*linspace(0,1,n_a)'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

% Discretize AR(1) using Farmer-Toda method
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);

%% Permanent types: values and masses

% Assume the fixed-effect alpha_i is normally distributed
% Discretize fixed-effect using Farmer-Toda method
[alpha_grid,pi_alpha]=discretizeAR1_FarmerToda(0,0,Params.sigma_alpha,N_i);
pi_alpha=pi_alpha(1,:)'; % i.i.d, so keep first row as a column

% The grid is the values for alpha, so we just put this in Params
Params.alpha_i=alpha_grid; % Because this parameter has a dimension of length N_i it will be automatically interpreted as depending on permanent type

% The dist we put in the parameter structure
Params.alpha_dist=pi_alpha;
% And we tell VFI Toolkit the name of the parameter which is the masses of the permanent types
PTypeDistParamNames={'alpha_dist'};

%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta','sj'};

% ReturnFn
ReturnFn=@(h,aprime,a,z,sigma,psi,eta,w,r,kappa_j,alpha_i,agej, Jr)...
    WorkshopModel5_ReturnFn(h,aprime,a,z,sigma,psi,eta,w,r,kappa_j,alpha_i,agej, Jr);
% First inputs are 'action space', here (h,aprime,a,...), everything
% after this is interpreted as a parameter.
% Note: alpha_i depending on permanent type i is handled internally by VFI Toolkit

%% Solve for value function and policy function
% Use both divide-and-conquer and grid interpolation layer
vfoptions.divideandconquer=1;
vfoptions.gridinterplayer=1;
vfoptions.ngridinterp=30;
tic;
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j, N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc
% Note: use _PType() version, which also means different inputs

% When using grid interpolation layer, have to tell simoptions too
simoptions.gridinterplayer=vfoptions.gridinterplayer;
simoptions.ngridinterp=vfoptions.ngridinterp;

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,ceil(n_z/2))=1; % start with 0 assets, median z shock
% You can alternatively set jequaloneDist=zeros([n_a,n_z,N_i],'gpuArray');

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationary Distribution
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);
% Note: use _PType() version, which also means different inputs

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j,alpha_i) w*kappa_j*h*exp(alpha_i+z); % the labor earnings
FnsToEvaluate.assets=@(h,aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate.alpha=@(h,aprime,a,z,alpha_i) exp(alpha_i); % fixed-effect

% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Note how we use names earnings and assets

%% Calculate various stats
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Params, n_d, n_a, n_z, N_j, N_i, d_grid, a_grid, z_grid, simoptions);
% Note: use _PType() version, which also means different inputs

% for example,
fprintf('Aggregate (Mean) earnings=%1.4f \n', AllStats.earnings.Mean) % Note: mass of agents is one, hence aggregate and mean are the same thing
fprintf('Gini coeff of assets=%1.2f \n', AllStats.assets.Gini)

% Note how it uses the names we gave the FnsToEvaluate

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,simoptions);
% Note: use _PType() version, which also means different inputs

figure(1);
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
title('Age-conditional mean of earnings')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.assets.Gini)
title('Age-conditional Gini coeff of assets')
% Doing much the same as AllStats, except now it is conditional on age
% Note: now people differ by age

%% Results for model statistics include but conditional and unconditional on permanent type
% Notice how we have both conditional-on-ptype
AgeConditionalStats.alpha.Mean
%  and unconditional-on-ptype
expalpha_i=[AgeConditionalStats.alpha.ptype001.Mean; AgeConditionalStats.alpha.ptype002.Mean; AgeConditionalStats.alpha.ptype003.Mean; AgeConditionalStats.alpha.ptype004.Mean; AgeConditionalStats.alpha.ptype005.Mean]
% Since this stat is simple, notice how the unconditional is just
sum(expalpha_i.*pi_alpha,1)

% Above is a bit silly, as the fixed-effect is independent of age, so gives
% same value for all ages. But we can see it is all correct from looking at
exp(Params.alpha_i)
% which gives the conditional-on-ptype result and at
sum(exp(Params.alpha_i).*pi_alpha)
% which gives the 'unconditional' result
