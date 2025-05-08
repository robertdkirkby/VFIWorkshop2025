% Workshop Exercise 2
% Modify WorkshopModel4 to remove the z (markov) shock.
% So solving a model with an e (i.i.d.) shock, but no z (markov) shock.

% Changes are
% Set n_z=0
% [Deleted parameters relating to z process, but this is not important, just cleaning]
% Set z_grid=[], pi_z=[]
% Modify ReturnFn to remove z from budget constraint
% Modify ReturnFn and FnsToEvaluate for new action space: (h,aprime,a,e,...)
% Modify jequaloneDist to remove z
% Done!


%% Model action and state-space
n_d=21; % number of grid points for our decision variable, labor supply
n_a=201; % number of grid points for our endogenous state, assets
n_z=0; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
n_e=7; % number of grid poitns for our exogenous i.i.d state, also labor productivity
N_j=81; % periods, represent ages 20 to 100

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

% Exogenous i.i.d. on labor productivity per time worked
Params.sigma_e=0.3;

% Conditional survival probabilities
Params.sj=[ones(1,46),linspace(1,0.99,81-46-10),linspace(0.99,0.9,9),0];

%% Grids

d_grid=linspace(0,1,n_d)'; % note, implicitly imposes the 0<h<1 constraint

a_grid=10*linspace(0,1,n_a)'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

% Discretize AR(1) using Farmer-Toda method
z_grid=[];
pi_z=[];

% Discretize Normal dist using Farmer-Toda method
[e_grid,pi_e]=discretizeAR1_FarmerToda(0,0,Params.sigma_e,n_e);
pi_e=pi_e(1,:)'; % use first row, as a column vector [i.i.d. probability dist is the 'conditional transition probabilities']

% Using e
vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;


%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta','sj'};

% ReturnFn
ReturnFn=@(h,aprime,a,e,sigma,psi,eta,w,r,kappa_j,agej, Jr)...
    WorkshopExercise2_ReturnFn(h,aprime,a,e,sigma,psi,eta,w,r,kappa_j,agej, Jr);
% First inputs are 'action space', here (h,aprime,a,e,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
% Note: we have moved beyond the basic features, so VFI Toolkit gets told about 'e' 
% variables by vfoptions (and simoptions), and hence the inputs are unchanged.
vfoptions.divideandconquer=1; % exploits monotonicity
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_e],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,:)=shiftdim(pi_e,1); % start with 0 assets, and the dist of e

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationart Distribution
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(h,aprime,a,e,w,kappa_j) w*kappa_j*h*exp(e); % the labor earnings
FnsToEvaluate.assets=@(h,aprime,a,e) a; % a is the current asset holdings
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


