% Workshop Model 1

% Seven core steps:
% 1. Model action and state-spaces.
% 2. Parameters
% 3. Grids
% 4. ReturnFn
% 5. Solve for value fn and policy fn.
% 6. Agents stationary distribution.
% 7. Generate model moments/statistics.




%% Model action and state-space
n_d=0;
n_a=201; % number of grid points for our endogenous state, assets
n_z=0;
N_j=81; % periods, represent ages 20 to 100

%% Parameters

% Age and Retirement
Params.J=N_j; % final period
Params.agej=1:1:N_j; % model period
Params.agejshifter=19; % add to agej to get actual age
Params.Jr=65-19; % retire at age 65, which is period 46

% Preferences
Params.beta=0.98; % discount factor
Params.sigma=2; % CRRA utilty fn

% Prices
Params.r=0.05;
Params.w=1; % wage


% Deterministic earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% hump-shaped, then zero in retirement

% Conditional survival probabilities
Params.sj=[ones(1,46),linspace(1,0.99,81-46-10),linspace(0.99,0.9,9),0];

%% Grids

a_grid=10*linspace(0,1,n_a)'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

% We are not using them so,
d_grid=[];
z_grid=[];
pi_z=[];


%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta','sj'};

% ReturnFn
ReturnFn=@(aprime,a,sigma,w,r,kappa_j,agej, Jr)...
    WorkshopModel1_ReturnFn(aprime,a,sigma,w,r,kappa_j,agej, Jr);
% First inputs are 'action space', here (aprime,a,...), everything
% after this is interpreted as a parameter.
% Note: kappa_j depending on 'age' j is handled internally by VFI Toolkit


%% Solve for value function and policy function
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros(n_a,1,'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1)=1; % start with 0 assets

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationart Distribution
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(aprime,a,w,kappa_j) w*kappa_j; % w*kappa_j is the labor earnings
FnsToEvaluate.assets=@(aprime,a) a; % a is the current asset holdings
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Note how we use names earnings and assets

%% Calculate various stats
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);

% for example,
fprintf('Aggregate (Mean) earnings=%1.4f \n', AllStats.earnings.Mean) % Note: mass of agents is one, hence aggregate and mean are the same thing
fprintf('Gini coeff of assets=%1.2f \n', AllStats.assets.Gini)

% Note how it uses the names we gave the FnsToEvaluate

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% for example
figure(1);
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
title('Age-conditional mean of earnings')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.assets.Gini)
title('Age-conditional Gini coeff of assets')
% Doing much the same as AllStats, except now it is conditional on age
% Note: everyone in this model is identical (conditional on age), hence gini is 0

