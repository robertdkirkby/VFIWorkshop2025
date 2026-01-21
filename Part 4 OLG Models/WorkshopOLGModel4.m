% Workshop OLG Model 4: Calibrate OLG Model

% Model is same as Workshop OLG Model 1, and largely just copy-paste.
% Had to make some small changes so that w is now treated as a parameter to
% be determined in general eqm. (This was necessary as one of the
% calibration targets, namely age-conditional earnings, depends on the wage
% w, so either I had to replace w with the formula based on r that we used
% in the ReturnFn in Workshop OLG Model 1, which would be better but I was
% too lazy/looked messy, or I just made w determined by a general eqm condition 
% and then use it as an input to the ReturnFn.)

% Skip to line 167ish for the calibration.


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
Params.r=0.05; % initial guess for interest rate
Params.w=1; % initial guess for wage

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
ReturnFn=@(h,aprime,a,z,sigma,psi,eta,r,w,kappa_j,agej, Jr)...
    WorkshopOLGModel4_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,w,kappa_j,agej, Jr);
% First inputs are 'action space', here (h,aprime,a,...), everything
% after this is interpreted as a parameter.

%% Solve for value function and policy function
vfoptions.divideandconquer=1; % Use divide-and-conquer to exploit conditional monotonicity
vfoptions.gridinterplayer=1; % Use grid interpolation layer
vfoptions.ngridinterp=20; % aprime includes 20 evenly spaced points between every pair of a_grid points
simoptions.gridinterplayer=vfoptions.gridinterplayer; % tell simoptions about grid interpolation layer
simoptions.ngridinterp=vfoptions.ngridinterp;

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

% Solve Stationary Distribution
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% MOST OF THE CODE BEFORE THIS IS UNCHANGED
% Except small changes to Params and ReturnFn

%% Set up FnsToEvaluate
FnsToEvaluate.L=@(h,aprime,a,z,kappa_j) kappa_j*h*exp(z); % effective labor supply
FnsToEvaluate.K=@(h,aprime,a,z) a; % assets
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Just make sure they are working okay
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);


%% General Eqm
GEPriceParamNames={'r','w'};
% note, Params.r we set earlier was an inital guess

GeneralEqmEqns.capitalmarket=@(r,alpha,delta,K,L) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta);
GeneralEqmEqns.labourmarket=@(w,alpha,K,L) w-(1-alpha)*(K^(alpha))*(L^(-alpha));
% GeneralEqmEqn inputs must be either Params or AggVars (AggVars is like AllStats, but just the Mean)
% So here: r, alpha, delta will be taken from Params, and K,L will be taken from AggVars

%% Solve for stationary general eqm
heteroagentoptions.verbose=1; % just use defaults
tic;
[p_eqm,GEcondns]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
toc
% Done, the general eqm prices are in p_eqm
% GEcondns tells us the values of the GeneralEqmEqns, should be near zero
fprintf('General eqm was found with r=%1.3f \n', p_eqm.r)

% To be able to analyze the general eqm, we need to use the r we found
Params.r=p_eqm.r;
Params.w=p_eqm.w;

%% Evaluate the general eqm
% now that we have the general eqm, this is just like how we evaluated the life-cycle models.
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% add another FnsToEvaluate
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*h*exp(z); % w*kappa_j is the labor earnings
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z,w,kappa_j) h; % fraction of time worked


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


%% Now for the calibration
% We will use the results of the above model as the targets, so we know
% what the 'true' parameter values are. Let's replace them with some
% initial guesses so that we are not starting from the true values
Params.beta=0.96;
Params.sigma=3;
Params.psi=2;
% Same for the general eqm parameter, change it to something other than the solution
Params.r=0.05;
Params.w=1.5;

%%
CalibParamNames={'beta','sigma','psi'};

TargetMoments.AgeConditionalStats.earnings.Mean=AgeConditionalStats.earnings.Mean; % Set target for age-conditional mean earnings (this would normally be a target value that comes from data)
TargetMoments.AgeConditionalStats.fractiontimeworked.Mean=AgeConditionalStats.fractiontimeworked.Mean; % Set target for age-conditional mean fraction-of-time-worked (this would normally be a target value that comes from data)

caliboptions=struct(); % Just use the defaults
caliboptions.contrainpositive={'psi'}; % Not actually needed here, just to demonstrate
caliboptions.relativeGEweight=10; % This is the default, it applies a 'weight' of 10 to the general eqm eqns, relative to the 'weight' for the calibration targets [GE must hold, while Calibration targets are 'nice to hold', so should be notably above 1]
% Note: this is on top of giving different weights to the different GE eqns
% with heteroagentoptions.multiGEweights, and giving different weights to
% the different calibration targets with caliboptions.weights

[CalibParams,calibsummary]=CalibrateOLGModel(CalibParamNames,TargetMoments,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames, [], GEPriceParamNames, FnsToEvaluate, GeneralEqmEqns, heteroagentoptions, caliboptions, vfoptions,simoptions);

% Look at result
CalibParams
% Note: we know the true values should be
% beta=0.98
% sigma=2
% psi=3
% r=0.026
% w=1.53
% Can see we have found essentially the same

% Update Params based on results
for pp=1:length(CalibParamNames)
    Params.(CalibParamNames{pp})=CalibParams.(CalibParamNames{pp});
end
for pp=1:length(GEPriceParamNames)
    Params.(GEPriceParamNames{pp})=CalibParams.(GEPriceParamNames{pp});
end

% While CalibParams gives you the general eqm parameters that it found
% you should still next re-run the general eqm command as you want it to hold
% exactly, whereas the CalibrateOLGModel command by default does joint-optimization
% and so general eqm was balanced against calibration.

%% Analyse results our of calibration
[p_eqm2,GEcondns2]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% Update parameters
Params.r=p_eqm2.r;
Params.w=p_eqm2.w;

[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);
AgeConditionalStats2=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);


figure(2);
subplot(3,1,1); plot(Params.agejshifter+Params.agej, AgeConditionalStats.earnings.Mean)
hold on
plot(Params.agejshifter+Params.agej, AgeConditionalStats2.earnings.Mean)
hold off
title('Age-conditional mean of earnings')
legend('target','calibrated')
subplot(3,1,2); plot(Params.agejshifter+Params.agej, AgeConditionalStats.fractiontimeworked.Mean)
hold on
plot(Params.agejshifter+Params.agej, AgeConditionalStats2.fractiontimeworked.Mean)
hold off
legend('target','calibrated')
title('Age-conditional mean of fraction-time-worked')
subplot(3,1,3); plot(Params.agejshifter+Params.agej, AgeConditionalStats2.K.Mean)
title('Age-conditional mean of assets')
legend('calibrated (not a target)')
% Doing much the same as AllStats, except now it is conditional on age
% Note: now people differ by age