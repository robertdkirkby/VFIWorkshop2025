% Workshop OLG Model 2

% Add some government (pension plus an earnings tax; progressive income tax
% plus trasfers) which involves two more general eqm eqns.
% Also add a calibration target as a general eqm eqn (and give it a lower
% weight than the other general eqm eqns).

% We could/should use same 'trick' for w here as was used in previous model.
% But I just use w directly, as it makes my formulas for 'earnings' and
% 'income' much easier (in FnsToEvaluate). And solving the model is easy
% enough anyway. Sometimes, the lazy options is the better option :)

% Note: increased n_d and n_a as otherwise won't be able to hit the
% General Eqm Eqns to the desired accuracy.

%% Model action and state-space
n_d=51; % number of grid points for our decision variable, labor supply
n_a=301; % number of grid points for our endogenous state, assets
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
N_j=81; % periods, represent ages 20 to 100% 

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

% Taxes 
Params.tau_e=0.1; % tax on earnings
Params.tau_i1=0.1; % level of income tax
Params.tau_i2=0.7; % progressivity of income tax

% Pensions
Params.pension=0.4;
% Government spending
Params.G=0.5;

% Prices
Params.r=0.05; % interest rate
Params.w=1; % wage

% Firm problem
Params.alpha=0.36;
Params.delta=0.05; % depreciation rate

% Deterministic earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% hump-shaped, then zero in retirement

% Exogenous AR(1) on labor productivity per time worked
Params.rho_z=0.9; % autocorrelation coefficient
Params.sigma_z_epsilon=0.1; % std dev of innovations

% Conditional survival probabilities
Params.sj=[ones(1,46),linspace(1,0.99,81-46-10),linspace(0.99,0.9,9),0];

% Calibration target
Params.KdivYcalibtarget=3; % capital-output ratio (for use as calibration target)

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
ReturnFn=@(h,aprime,a,z,sigma,psi,eta,r,w,kappa_j,agej, Jr, tau_e, tau_i1, tau_i2, pension)...
    WorkshopOLGModel2_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,w,kappa_j,agej, Jr, tau_e, tau_i1, tau_i2, pension);
% First inputs are 'action space', here (h,aprime,a,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
vfoptions.divideandconquer=1; % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Agent distribution

% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,ceil(n_z/2))=1; % start with 0 assets, median z shock

% Mass of agents of each age
Params.mewj=ones(N_j,1)/N_j; % equal mass of each age (must some to one)
AgeWeightParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
% Note: should set mewj based on sj, but this is just a very simple example

% Solve Stationart Distribution
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% Set up FnsToEvaluate
FnsToEvaluate.L=@(h,aprime,a,z,kappa_j) kappa_j*h*exp(z); % effective labor supply
FnsToEvaluate.K=@(h,aprime,a,z) a; % assets
FnsToEvaluate.earningstaxrevenue=@(h,aprime,a,z,w,kappa_j,tau_e) tau_e*w*kappa_j*h*exp(z); % tau_e*earnings
FnsToEvaluate.incometaxrevenue=@(h,aprime,a,z,r,w,kappa_j,tau_i1,tau_i2) (r*a+w*kappa_j*h*exp(z))*(1-tau_i1*(r*a+w*kappa_j*h*exp(z))^tau_i2); % income*(1-tau_i1*income^tau_i2)
FnsToEvaluate.pensionspending=@(h,aprime,a,z,pension,agej,Jr) pension*(agej>=Jr); % pension is received by retirees
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Just make sure they are working okay
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);


%% General Eqm
GEPriceParamNames={'w','r','pension','G','beta'};
% note, Params.r we set earlier was an inital guess (all these must already be in Params)

GeneralEqmEqns.labormarket=@(w,alpha,K,L) w-(1-alpha)*(K^alpha)*(L^(-alpha));
GeneralEqmEqns.capitalmarket=@(r,alpha,delta,K,L) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta);
GeneralEqmEqns.pensionbalance=@(earningstaxrevenue,pensionspending) earningstaxrevenue-pensionspending;
GeneralEqmEqns.govbudgetbalance=@(G, incometaxrevenue) G-incometaxrevenue;
GeneralEqmEqns.KdivYtarget=@(alpha,K,L,KdivYcalibtarget) K/((K^(alpha))*(L^(1-alpha)))-KdivYcalibtarget;
% GeneralEqmEqn inputs must be either Params or AggVars (AggVars is like AllStats, but just the Mean)
% So here: r, alpha, delta will be taken from Params, and K,L will be taken from AggVars

% Note: there is no need to have GEPriceParamNames and GeneralEqmEqns
% somehow 'in the same order' from the perspective of the codes. But I find 
% that when I am creating models it helps me think about them as being a 
% 'price per eqm eqn' and writing them out in the same order. Makes it
% easier to then add/remove things and modify the model later.

%% Solve for stationary general eqm
heteroagentoptions.multiGEweights=[1,1,1,1,0.5]; % Note: look at GeneralEqmEqns to see the order that these relate to
heteroagentoptions.verbose=1; % just use defaults
[p_eqm,GEcondns]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% Done, the general eqm prices are in p_eqm
% GEcondns tells us the values of the GeneralEqmEqns, should be near zero

% So general eqm was
p_eqm
% and we can see we 'hit' all the eqns (they are essentially zero)
GEcondns

% To be able to analyze the general eqm, we need to use the general eqm
% params we found
for pp=1:length(GEPriceParamNames)
    Params.(GEPriceParamNames{pp})=p_eqm.(GEPriceParamNames{pp});
end

%% Evaluate the general eqm
% now that we have the general eqm, this is just like how we evaluated the life-cycle models.
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% add another FnsToEvaluate
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*h*exp(z); % w*kappa_j is the labor earnings


%% Calculate various stats
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);

% for example,
fprintf('Aggregate (Mean) earnings=%1.4f \n', AllStats.earnings.Mean) % Note: mass of agents is one, hence aggregate and mean are the same thing
fprintf('Gini coeff of assets=%1.2f \n', AllStats.K.Gini)

% Note how it uses the names we gave the FnsToEvaluate

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);


figure(1);
subplot(2,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
title('Age-conditional mean of earnings')
subplot(2,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.K.Mean)
title('Age-conditional mean of assets')
% Doing much the same as AllStats, except now it is conditional on age
% Note: now people differ by age


