% Workshop Model 7
% Further model analysis

% Model is just that from Workshop Model 3, skip to line 117
% Further model analysis starts there



%% Model action and state-space
n_d=21; % number of grid points for our decision variable, labor supply
n_a=201; % number of grid points for our endogenous state, assets
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
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


%% Okay, now let's do so further analysis of this model

%% We might want to get the values of the optimal policy (rather than the index, which is what Policy contains)
PolicyVals=PolicyInd2Val_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions);

%% Simulate some panel data, and run a regression on the results
% Note: simulated panels take longer runtime than StationaryDist, hence why we previously just use StationaryDist.

% Panel data simulation uses FnsToEvaluate
simoptions.numbersims=1000;
SimPanelData=SimPanelValues_FHorz_Case1(jequaloneDist, Policy, FnsToEvaluate, Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z,simoptions);

% Run a regression of earnings on it's own lag (essentially, estimate the autocorrelation of earnings)
y=SimPanelData.earnings(:,2:end); % first dimension is the cross-section, second dimension is the finite-horizon period
ylag=SimPanelData.earnings(:,1:end-1);
[b,bint,~,~,regstats]=regress(y(:),[ones(size(ylag(:))),ylag(:)]); % regress earnings on constant and own lag

% clean simoptions before we go to next thing
simoptions=rmfield(simoptions,'numbersims');

%% Calculate conditional statistics based on a restriction

% Create a restriction [same setupt as FnsToEvaluate, but will evaluate to 1 for satifying restriction, 0 for violating restriction]
simoptions.conditionalrestrictions.zeroassets=@(h,aprime,a,z) (a==0);
% Just to see that we can, let's do two separate restrictions
Params.avgearnings=AllStats.earnings.Mean;
simoptions.conditionalrestrictions.aboveavgearnings=@(h,aprime,a,z,w,kappa_j,avgearnings) (w*kappa_j*h*exp(z))>avgearnings;
% [To impose two joint-restrictions you would need to set up one function that imposed both the restrictions]

FnsToEvaluate.satisfyzeroassets=simoptions.conditionalrestrictions.zeroassets; % we will use this to measure fraction of people satisfying a restriction 

FnsToEvaluate.mass=@(h,aprime,a,z) 1; % this is silly, but just want to use it to emphasize when stats are conditional


% When we call AllStats or LifeCycleProfiles it will also include the values based on these conditional restrictions
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example, the assets of people with zero assets [silly, but we know the answer so easy way to see what is happening]
AllStats.zeroassets.assets.Mean
% Or we can look at fraction of people with zero assets at each age
AllStats.satisfyzeroassets.Mean

% Notice that if we look at
AllStats.zeroassets.mass.Mean
% we get that it is 1, which is just emphasizing that these stats are conditional


% Or at the asset holdings of people with earnings above the mean (conditional on age)
AgeConditionalStats.aboveavgearnings.assets.Mean
% note the asset holdings are conditional on age and on having earnings above the mean earnings, but this mean earnings cutoff is for the whole population
% note, obviously no-one who is retired has earnings above the average
% earnings, when no-one satisfies the restriction you get NaN, as seen here

% clean simoptions before we go to next thing
simoptions=rmfield(simoptions,'conditionalrestrictions');

%% Calculate stats faster (if you don't need them all)

% normally we calculate all the stats
simoptions.whichstats=ones(7,1); % default: calculate them all
tic;
AllStats1=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);
time1=toc

% We can skip some if we need speed (useful if you are writing your own calibration/estimation commands for something)
simoptions.whichstats(4:7)=0; % skip the last few [search forum to find details: discourse.vfitoolkit.com ]
tic;
AllStats2=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);
time2=toc

% Look at AllStats1 and AllStats2, notice how the later is missing a bunch of stats
% Look at time1 and time2, notice how the later is faster (smaller runtime)
% The two slowest stats to compute are the lorenz curve and the quantiles.

% clean simoptions before we go to next thing
simoptions=rmfield(simoptions,'whichstats');

%% Details for Lorenz curve and quantiles
% Number of points used for Lorenz Curve
simoptions.npoints=100;
% Number of points used for Quantiles (means and cutoffs)
simoptions.nquantiles=20;
% These work for both AllStats and AgeConditionalStats

%% Group ages for age-conditional stats: e.g., 5-year age bins

% default is 
simoptions.agegroupings=1:1:N_j; % model period at which each 'age-group' starts
AgeConditionalStats_1yr=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% We could set 5-year age bins
simoptions.agegroupings=1:5:N_j; % model period at which each 'age-group' starts
AgeConditionalStats_5yr=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% We could set working age and retirement
simoptions.agegroupings=[1,Params.Jr]; % model period at which each 'age-group' starts (Jr is first period of retirement)
AgeConditionalStats_WorkRet=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% Look at sizes of each
AgeConditionalStats_1yr.earnings.Mean
AgeConditionalStats_5yr.earnings.Mean
AgeConditionalStats_WorkRet.earnings.Mean
% Because these are just means we could easily calculate the 5yr from the
% 1yr together with the age weights, but these commands are powerful for
% things like
% Gini of earnings for working age population
AgeConditionalStats_WorkRet.earnings.Gini(1) % note: 1 is the first of the two age groupings, which relates to working age (ages 1 to Jr-1, inclusive)

% clean simoptions before we go to next thing
simoptions=rmfield(simoptions,'agegroupings');


%% Doing custom model outputs
% One option is to use panel data, as we saw above
% Internally, e.g. in AllStats, the toolkit uses the stationary dist together with the values of the FnsToEvaluate on the grid.
% We can get these using

ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);



