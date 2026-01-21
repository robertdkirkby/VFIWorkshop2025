% Workshop Model 16
% Portfolio-Choice: riskyasset

% Tell vfoptions we are using riskyasset
% Set up vfoptions.aprimeFn which is aprime(d,u)
% Action space no longer contains aprime (as it cannot be chosen directly)

% Note: Linear interpolation is used to put aprime(d,u) onto the a_grid, so we can use less points for a_grid (less points for any given desired accuracy level)

% If you have multiple endogenous states, the riskyasset will be the 'last' one.
% riskyasset uses 'refine_d' which means that not all d are in ReturnFn and aprimeFn (allows code to be much faster)
% Note: all decision variables are still in action space, so still used for FnsToEvaluate

%% Model action and state-space
n_d=[51,201]; % number of grid points for our decision variable, risky share, savings
n_a=201; % number of grid points for our endogenous state, assets
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
n_u=7; % number of grid points for i.i.d. between-period shock, risky asset return
N_j=81; % periods, represent ages 20 to 100

% Note: that d2 and a use same number of grid points is not necessary

% Note: because u is between-periods it will not actually be part of the
% action space nor of the state space. But I declare it here anyway.


%% To speed up the use of riskyasset we use 'refine_d', which requires us to set the decision variables in a specific order
vfoptions.refine_d=[0,1,1]; % tell the code how many d1, d2, and d3 there are
% Idea is to distinguish three categories of decision variable:
%  d1: decision is in the ReturnFn but not in aprimeFn
%  d2: decision is in the aprimeFn but not in ReturnFn
%  d3: decision is in both ReturnFn and in aprimeFn
% Note: ReturnFn must use inputs (d1,d3,..) 
%       aprimeFn must use inputs (d2,d3,..)
% n_d must be set up as n_d=[n_d1, n_d2, n_d3]
% d_grid must be set up as d_grid=[d1_grid; d2_grid; d3_grid];
% It is possible to solve models without any d1, as is the case here.
simoptions.refine_d=vfoptions.refine_d;

%% Parameters

% Age and Retirement
Params.J=N_j; % final period
Params.agej=1:1:N_j; % model period
Params.agejshifter=19; % add to agej to get actual age
Params.Jr=65-19; % retire at age 65, which is period 46

% Preferences
Params.beta=0.98; % discount factor
Params.sigma=2; % curvature of consumption in utility fn

% Prices
Params.r=0.05; % interest rate on safe asset
Params.w=1; % wage

% Risky asset
Params.riskpremium=0.02;
Params.sigma_u=0.02;

% Deterministic earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% hump-shaped, then zero in retirement

% Exogenous AR(1) on labor productivity per time worked
Params.rho_z=0.9; % autocorrelation coefficient
Params.sigma_z_epsilon=0.1; % std dev of innovations

% Conditional survival probabilities
Params.sj=[ones(1,46),linspace(1,0.99,81-46-10),linspace(0.99,0.9,9),0];

%% Grids

riskyshare_grid=linspace(0,1,n_d(1))'; % fraction of assets invested in risky asset (rest is safe asset)

assets_grid=10*linspace(0,1,n_a)'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

% Discretize AR(1) using Farmer-Toda method
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);

% put in tookit form
d_grid=[riskyshare_grid; assets_grid];
a_grid=assets_grid;
% Note: that d2 and a use same grid is not necessary

%% riskyasset: aprimeFn
% To use an riskyasset, we need to define aprime(d,u)
% [in notation of the current model, aprime(d1,d2,u)]

vfoptions.riskyasset=1; % Using a risky asset
% Note: by default, assumes it is the last d variable that controls the
% evolution of the experience asset u (and that the last a variable is
% the risky asset).

% aprimeFn gives the value of hprime
vfoptions.aprimeFn=@(riskyshare,savings,u,r,riskpremium) savings*(1+(1-riskyshare)*r+riskyshare*(r+riskpremium+u));
% The first inputs must be (d,u) [in the sense of aprime(d,u)], then any parameters

% We also need to tell simoptions about the riskyasset
simoptions.riskyasset=1;
simoptions.aprimeFn=vfoptions.aprimeFn;
simoptions.d_grid=d_grid; % Needed to handle aprimeFn
simoptions.a_grid=a_grid; % Needed to handle aprimeFn

% We need to provide the i.i.d. u [n_u, u_grid, pi_u]
[u_grid,pi_u]=discretizeAR1_FarmerToda(0,0,Params.sigma_u,n_u);
pi_u=pi_u(1,:)'; % u is iid
% Put into vfoptions and simoptions
vfoptions.n_u=n_u;
vfoptions.u_grid=u_grid;
vfoptions.pi_u=pi_u;
simoptions.n_u=vfoptions.n_u;
simoptions.u_grid=vfoptions.u_grid;
simoptions.pi_u=vfoptions.pi_u;

% Note: I use u~N(0,sigma_u^2)
% Could equivalently do u~N(r+riskpremium, sigma_u^2) instead (with corresponding change to aprimeFn)

%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta','sj'};

% ReturnFn
ReturnFn=@(savings,a,z,sigma,w,kappa_j,agej, Jr)...
    WorkshopModel16_ReturnFn(savings,a,z,sigma,w,kappa_j,agej, Jr);
% First inputs are 'action space', here (savings,a,z,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
% Cannot use divide-and-conquer, nor grid interpolation layer on a 'riskyasset'  [if there were two endogenous states, we could use them on the other]
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,ceil(n_z/2))=1; % start with 0 assets, median z shock

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationart Distribution
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(riskyshare,savings,a,z,w,kappa_j) w*kappa_j*exp(z); % the labor earnings
FnsToEvaluate.assets=@(riskyshare,savings,a,z) a; % a is the current asset holdings
FnsToEvaluate.riskyshare=@(riskyshare,savings,a,z) riskyshare; % a is the current asset holdings
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
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.riskyshare.Mean)
title('Age-conditional mean of riskyshare')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.assets.Mean)
title('Age-conditional mean of assets')
% Doing much the same as AllStats, except now it is conditional on age
% Note: now people differ by age


