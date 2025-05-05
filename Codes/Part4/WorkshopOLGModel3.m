% Workshop OLG Model 3

% Permanent types: N_i=5, they are different fixed-effect in earnings
% And bequests are conditional on permanent type
% In General Eqm we need 'bequests received' equals 'bequests left', and
% this will be conditional on permanent type.


%% Model action and state-space
n_d=21; % number of grid points for our decision variable, labor supply
n_a=201; % number of grid points for our endogenous state, assets
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
N_j=81; % periods, represent ages 20 to 100% 

N_i=5; % fixed-effect in earnings (five quintiles)

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
Params.r=0.05; % initial guess for interest rate

% Bequests
Params.Jbeq1=40-Params.agejshifter; % Receive beq from age 40...
Params.Jbeq2=60-Params.agejshifter; % ...to age 60
Params.Beq=[0.05,0.1,0.15,0.2,0.25]; % bequests, by permanent type (just an initial guess)

% Firm problem
Params.alpha=0.36;
Params.delta=0.05; % depreciation rate

% Deterministic earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% hump-shaped, then zero in retirement
% Fixed-effect in earnings
Params.gamma_i=[0.3,0.7,1,1.5,3];

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

% We are not using them so,
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);

%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta','sj'};

% ReturnFn
ReturnFn=@(h,aprime,a,z,sigma,psi,eta,r,alpha,delta,kappa_j,gamma_i,Beq,agej, Jr, Jbeq1, Jbeq2)...
    WorkshopOLGModel3_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,alpha,delta,kappa_j,gamma_i,Beq,agej, Jr, Jbeq1, Jbeq2);
% First inputs are 'action space', here (h,aprime,a,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
vfoptions.divideandconquer=1; % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc

%% Agent distribution

PTypeDistParamNames={'quintilemasses'};
Params.quintilemasses=[0.2,0.2,0.2,0.2,0.2];

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,ceil(n_z/2))=1; % start with 0 assets, median z shock

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationart Distribution
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.L=@(h,aprime,a,z,kappa_j) kappa_j*h*exp(z); % effective labor supply
FnsToEvaluate.K=@(h,aprime,a,z) a; % assets
FnsToEvaluate.BeqReceived=@(h,aprime,a,z,Beq,agej,Jbeq1,Jbeq2) Beq*(agej>=Jbeq1)*(agej<=Jbeq2); % assets
FnsToEvaluate.BeqLeft=@(h,aprime,a,z,sj) aprime*(1-sj); % assets left when people die
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Just make sure they are working okay
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Params, n_d, n_a, n_z, N_j, N_i, d_grid, a_grid, z_grid,simoptions);


%% General Eqm
GEPriceParamNames={'r','Beq'};
% note, Params.r we set earlier was an inital guess

GeneralEqmEqns.capitalmarket=@(r,alpha,delta,K,L) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta);
GeneralEqmEqns.bequests=@(BeqReceived,BeqLeft) BeqReceived-BeqLeft;
% GeneralEqmEqn inputs must be either Params or AggVars (AggVars is like AllStats, but just the Mean)
% So here: r, alpha, delta will be taken from Params, and K,L will be taken from AggVars

heteroagentoptions.GEptype={'bequests'}; % solve the bequests GeneralEqmEqn conditional on permanent type

%% Solve for stationary general eqm
heteroagentoptions.verbose=1; % just use defaults
[p_eqm,~,GEcondns]=HeteroAgentStationaryEqm_Case1_FHorz_PType(n_d, n_a, n_z, N_j, N_i,[], pi_z, d_grid, a_grid, z_grid, jequaloneDist,ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, AgeWeightParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% Done, the general eqm prices are in p_eqm
% GEcondns tells us the values of the GeneralEqmEqns, should be near zero
fprintf('General eqm was found with r=%1.3f \n', p_eqm.r)

% To be able to analyze the general eqm, we need to use the r we found
Params.r=p_eqm.r;
Params.Beq=p_eqm.Beq;
% so we can use it later, also
Params.w=(1-Params.alpha)*((Params.r+Params.delta)/Params.alpha)^(Params.alpha/(1-Params.alpha));


%% Evaluate the general eqm
% now that we have the general eqm, this is just like how we evaluated the life-cycle models.
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);
% add another FnsToEvaluate
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j,gamma_i) w*kappa_j*gamma_i*h*exp(z); % w*kappa_j is the labor earnings


%% Calculate various stats
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Params, n_d, n_a, n_z, N_j,N_i, d_grid, a_grid, z_grid,simoptions);

% for example,
fprintf('Aggregate (Mean) earnings=%1.4f \n', AllStats.earnings.Mean) % Note: mass of agents is one, hence aggregate and mean are the same thing
fprintf('Gini coeff of assets=%1.2f \n', AllStats.K.Gini)

% Note how it uses the names we gave the FnsToEvaluate

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,simoptions);


figure(1);
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
title('Age-conditional mean of earnings')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.K.Mean)
title('Age-conditional mean of assets')
% Doing much the same as AllStats, except now it is conditional on age
% Note: now people differ by age

figure(2);
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype001.Mean)
hold on
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype002.Mean)
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype003.Mean)
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype004.Mean)
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.ptype005.Mean)
hold off
title('Age-conditional mean of earnings, by ptype')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.K.ptype001.Mean)
hold on
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.K.ptype002.Mean)
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.K.ptype003.Mean)
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.K.ptype004.Mean)
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.K.ptype005.Mean)
hold off
title('Age-conditional mean of assets, by ptype')
legend('ptype1','ptype2','ptype3','ptype4','ptype5')

