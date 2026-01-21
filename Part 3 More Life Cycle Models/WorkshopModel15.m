% Workshop Model 15
% Uncertain Human capital: experienceassetu

% Tell vfoptions we are using experienceassetu
% Set up vfoptions.aprimeFn which is aprime(d,a,u)
% Action space no longer contains aprime (as it cannot be chosen directly)

% Note: Linear interpolation is used to put aprime(d,a,u) onto the a_grid, so we can use less points for a_grid (less points for any given desired accuracy level)

% If you have multiple endogenous states, the experienceassetu will be the 'last' one.
% If you have multiple decision variables, the one used for experienceassetu will be the 'last' one

%% Model action and state-space
n_d=21; % number of grid points for our decision variable, labor supply
n_a=101; % number of grid points for our endogenous state, human capital
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
n_u=5; % number of grid points for our between-period i.i.d., shocks to human capital accumulation
N_j=45; % periods, represent ages 20 to 64 (just working age)

% Note: because u is between-periods it will not actually be part of the
% action space nor of the state space. But I declare it here anyway.

%% Parameters

% Age and Retirement
Params.J=N_j; % final period
Params.agej=1:1:N_j; % model period
Params.agejshifter=19; % add to agej to get actual age

% Preferences
Params.beta=0.98; % discount factor
Params.sigma=2; % curvature of consumption in utility fn

% Prices
Params.w=1; % wage

% Human capital accumulation
Params.ability=1;
Params.lscaler=0.2;
Params.alpha_h=0.5; % dimishing returns to human capital
Params.delta_h=0.1; % depreciation rate of human capital
Params.mean_u=0;
Params.sigma_u=0.1;

% Exogenous AR(1) on labor productivity per time worked
Params.rho_z=0.9; % autocorrelation coefficient
Params.sigma_z_epsilon=0.1; % std dev of innovations

%% Grids

d_grid=linspace(0,1,n_d)'; % l, labor supply

h_grid=5*linspace(0,1,n_a)'; % Column vector of length n_a

% Discretize AR(1) using Farmer-Toda method
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);

% renmae for toolkit
a_grid=h_grid;

%% experienceassetu: aprimeFn
% To use an experienceassetu, we need to define aprime(d,a,u)
% [in notation of the current model, hprime(l,h,u)]

vfoptions.experienceassetu=1; % Using an experience asset u
% Note: by default, assumes it is the last d variable that controls the
% evolution of the experience asset u (and that the last a variable is
% the experience asset u).

% aprimeFn gives the value of hprime
vfoptions.aprimeFn=@(l,h,u,alpha_h, delta_h, ability,lscaler) exp(u)*(ability*(h*(lscaler*(1-l))))^alpha_h+h*(1-delta_h);
% The first inputs must be (d,a,u) [in the sense of aprime(d,a,u)], then any parameters

% We also need to tell simoptions about the experienceassetu
simoptions.experienceassetu=1;
simoptions.aprimeFn=vfoptions.aprimeFn;
simoptions.d_grid=d_grid; % Needed to handle aprimeFn
simoptions.a_grid=a_grid; % Needed to handle aprimeFn

% We need to provide the i.i.d. u [n_u, u_grid, pi_u]
[u_grid,pi_u]=discretizeAR1_FarmerToda(Params.mean_u,0,Params.sigma_u,n_u);
pi_u=pi_u(1,:)'; % u is iid
% Put into vfoptions and simoptions
vfoptions.n_u=n_u;
vfoptions.u_grid=u_grid;
vfoptions.pi_u=pi_u;
simoptions.n_u=vfoptions.n_u;
simoptions.u_grid=vfoptions.u_grid;
simoptions.pi_u=vfoptions.pi_u;

%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta'};

% ReturnFn
ReturnFn=@(l,h,z,sigma,w)...
    WorkshopModel14_ReturnFn(l,h,z,sigma,w);
% First inputs are 'action space', here (l,h,z,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
% Cannot use divide-and-conquer, nor grid interpolation layer on a 'experienceassetu'  [if there were two endogenous states, we could use them on the other]
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(ceil(n_a/5),ceil(n_z/2))=1; % start with some human capital, median z shock

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationart Distribution
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(l,h,z,w) w*l*h*exp(z); % the labor earnings
FnsToEvaluate.humancapital=@(l,h,z) h; % h is the current human capital
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Note how we use names earnings and human capital

%% Calculate various stats
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);

% for example,
fprintf('Aggregate (Mean) earnings=%1.4f \n', AllStats.earnings.Mean) % Note: mass of agents is one, hence aggregate and mean are the same thing
fprintf('Gini coeff of human capital=%1.2f \n', AllStats.humancapital.Gini)

% Note how it uses the names we gave the FnsToEvaluate

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);


figure(1);
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
title('Age-conditional mean of earnings')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.humancapital.Mean)
title('Age-conditional mean of human capital')
% Doing much the same as AllStats, except now it is conditional on age
% Note: now people differ by age


