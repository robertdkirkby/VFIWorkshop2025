% Workshop Model 12
% Ambiguity Aversion

% Ambiguity involves p priors.
% So essentially only changes to the code are to say we want to use
% ambiguity aversion, and to set up the p priors.

% Only change is the section 'Set up priors for ambiguity'

%% Model action and state-space
n_d=21; % number of grid points for our decision variable, labor supply
n_a=201; % number of grid points for our endogenous state, assets
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
N_j=81; % periods, represent ages 20 to 100


%% Set up priors for ambiguity
N_z=prod(n_z);

vfoptions.exoticpreferences='AmbiguityAversion'; % say that we are using ambiguity aversion
vfoptions.n_ambiguity=3; % number of priors
% Now set up our three priors
% We are setting up a pi_z, of the same kind we would for any markov (so it
% will be a matrix of n_z-by-n_z).
vfoptions.ambiguity_pi_z=zeros(N_z,N_z,vfoptions.n_ambiguity); 
% the last dimension indexes the priors in our 'multiple priors'

% First prior,
vfoptions.ambiguity_pi_z(:,:,1)=ones(N_z,N_z)./N_z; % this prior is a uniform i.i.d.
% Second prior
[~,pi_z_temp]=discretizeAR1_FarmerToda(0,0.9,0.2,n_z); % AR(1) with rho=0.9, sigma_z_epsilon=0.2
vfoptions.ambiguity_pi_z(:,:,2)=pi_z_temp;
% Third prior
[~,pi_z_temp]=discretizeAR1_FarmerToda(0,0.7,0.1,n_z); % AR(1) with rho=0.7, sigma_z_epsilon=0.1
vfoptions.ambiguity_pi_z(:,:,3)=pi_z_temp;

% We still define the 'true' process for z below using z_grid and
% pi_z in the usual fashion. (Whether or not you use the 'true' as one of
% the priors or not is a modelling choice, here I don't.)
% The true process is an AR(1) with rho=0.9 and sigma_z_epsilon=0.1
% So the second and third priors are not miles from the truth, but none of
% our priors are the 'true' process, and the first prior is miles off.

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

a_grid=10*linspace(0,1,n_a)'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

% Discretize AR(1) using Farmer-Toda method
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);


%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta','sj'};

% ReturnFn
ReturnFn=@(h,aprime,a,z,sigma,psi,eta,w,r,kappa_j,agej, Jr)...
    WorkshopModel3_ReturnFn(h,aprime,a,z,sigma,psi,eta,w,r,kappa_j,agej, Jr);
% First inputs are 'action space', here (h,aprime,a,z,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
vfoptions.divideandconquer=1; % exploits monotonicity
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
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*h*exp(z); % the labor earnings
FnsToEvaluate.assets=@(h,aprime,a,z) a; % a is the current asset holdings
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


