% Workshop OLG Transition Path 3

% Multiple Reveals
% First reveal: In period 1, announce that tax will be cut in period 10.
% Second reveal: In period 8, announce that tax won't be cut until period 15.
% (period 15 relative to original period 1, not relative to start of second reveal path)

% Our reform:
tau_initial=0.1;
tau_final=0.05;

%% Model action and state-space
n_d=51; % number of grid points for our decision variable, labor supply
n_a=351; % number of grid points for our endogenous state, assets
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

% Prices
Params.r=0.05; % initial guess for interest rate (is determined in GE)
Params.w=1; % initial guess for wage (is determined in GE)

% Taxes:
Params.tau=0.1; % placeholder 
% Government spending
Params.G=0.2; % initial guess for gov spending (is determined in GE)

% Firm problem
Params.alpha=0.36;
Params.delta=0.05; % depreciation rate

% Deterministic earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% hump-shaped, then zero in retirement

% Exogenous AR(1) on labor productivity per time worked
Params.rho_z=0.9; % autocorrelation coefficient
Params.sigma_z_epsilon=0.1; % std dev of innovations


%% Grids

d_grid=linspace(0,1,n_d)'; % note, implicitly imposes the 0<h<1 constraint

a_grid=10*linspace(0,1,n_a)'.^3; % Column vector of length n_a
% ^3 will put more points near 0 than 1, model has more curvature here
% note, implicitly imposes the aprime>0 constraint

% We are not using them so,
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_z_epsilon,n_z);


%% ReturnFn
% Discount factors
DiscountFactorParamNames={'beta'};

% ReturnFn
ReturnFn=@(h,aprime,a,z,sigma,psi,eta,r,w,tau,kappa_j,agej, Jr)...
    WorkshopOLGTPath1_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,w,tau,kappa_j,agej, Jr);
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

%% MOST OF THE CODE BEFORE THIS IS UNCHANGED
% Except small changes to Params and ReturnFn

%% Set up FnsToEvaluate
FnsToEvaluate.L=@(h,aprime,a,z,kappa_j) kappa_j*h*exp(z); % effective labor supply
FnsToEvaluate.K=@(h,aprime,a,z) a; % assets
FnsToEvaluate.taxrevenue=@(h,aprime,a,z,w,tau,kappa_j) tau*w*kappa_j*h*exp(z); % effective labor supply
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Just make sure they are working okay
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);

% For analysing the model
FnsToEvaluate2=FnsToEvaluate;
FnsToEvaluate2.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*h*exp(z); % w*kappa_j is the labor earnings
% Note: I keep the FnsToEvaluate use in general eqm to a minimum (to reduce
% runtimes) and then use FnsToEvaluate2 to analyse model with more stats.

%% General Eqm
GEPriceParamNames={'r','w','G'};
% note, Params.r we set earlier was an inital guess

GeneralEqmEqns.capitalmarket=@(r,alpha,delta,K,L) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta); % r=marginal product of capital
GeneralEqmEqns.labormarket=@(w,alpha,K,L) w-(1-alpha)*(K^alpha)*(L^(-alpha)); % w=marginal product of labor
GeneralEqmEqns.govbudgetbalance=@(taxrevenue,G) taxrevenue-G;
% GeneralEqmEqn inputs must be either Params or AggVars (AggVars is like AllStats, but just the Mean)

heteroagentoptions.verbose=1; % just use defaults

%% Solve for initial stationary general eqm
Params.tau=tau_initial; % initial tax rate

[p_eqm_init,~,GEcondns_init]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% Done, the general eqm prices are in p_eqm
% GEcondns tells us the values of the GeneralEqmEqns, should be near zero

% To be able to analyze the general eqm, we need to use the r we found
Params.r=p_eqm_init.r;
Params.w=p_eqm_init.w;
Params.G=p_eqm_init.G;

% Evaluate the initial stationary general eqm
[V_init, Policy_init]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist_init=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy_init,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% Calculate various stats
AllStats_init=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist_init, Policy_init, FnsToEvaluate2, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);
% Calculate the life-cycle profiles
AgeConditionalStats_init=LifeCycleProfiles_FHorz_Case1(StationaryDist_init,Policy_init, FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% Note: Only part of this initial stationary general eqm we actually 'need'
% is the agent distribution. Rest is just out of interest.

AgentDist_init=StationaryDist_init; % Just to emphasize that there is no need for the
   % initial agent distribution to be a stationary dist (it is in this
   % example, but does not need to be for transition paths)

%% For multiple reveal, we don't need final stationary general eqm, as the multiple reveal command includes solving for them.
% %% Solve for final stationary general eqm
% Params.tau=tau_final; % final tax rate
% 
% [p_eqm_final,~,GEcondns_final]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% % Done, the general eqm prices are in p_eqm
% % GEcondns tells us the values of the GeneralEqmEqns, should be near zero
% 
% % To be able to analyze the general eqm, we need to use the r we found
% Params.r=p_eqm_final.r;
% Params.w=p_eqm_final.w;
% Params.G=p_eqm_final.G;
% 
% % Evaluate the initial stationary general eqm
% [V_final, Policy_final]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
% StationaryDist_final=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy_final,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% % Calculate various stats
% AllStats_final=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist_final, Policy_final, FnsToEvaluate2, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);
% % Calculate the life-cycle profiles
% AgeConditionalStats_final=LifeCycleProfiles_FHorz_Case1(StationaryDist_final,Policy_final, FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
% 
% % Note: Only part of this final stationary general eqm we actually 'need'
% % is the value fn (although we likely want p_eqm_final for initial guess of PricePath0). 
% % Rest is just out of interest.
% 
% % Double-check that the general eqm is accurate before we start the
% % transition path, because if it is not then it won't solve
% GEcondns_final

%%
% Can skip first part, as remains unchanged from Example 1
% load tpath1A.mat


%% Setup for the multiple-reveal transition path
T=100; % number of periods for transition path

% Setup ParamPath, roughly same as before but using 'tXXXX' for the
% period in which a path is revealed. We have a reveal in period 1, so that
% is
ParamPath.t0001.tau=[tau_initial*ones(1,9), tau_final*ones(1,T-9)]; % exogenous parameter path
% and then a second reveal in period 8, so
ParamPath.t0008.tau=[tau_initial*ones(1,7), tau_final*ones(1,T-7)]; % exogenous parameter path
% Note: in period 8, reveal that tax cut in period 15; viewed from period
% 8, the period 15 is the 8th period of the reveal (start counting from 8
% as being 1, then 9 is 2, ... , 15 is 8).


% With multiple reveal, rather than a guess for the path, we guess a
% 'shape' for the path (internally this 'shape' will go from initial to final values)
% Initial guess for general eqm parameters.
% Think of 0 as initial price, and 1 as final price. (Can set values outside of 0 to 1)
PricePathShaper.r=[zeros(1,7),linspace(0, 1,ceil(T/2)),ones(1,T-ceil(T/2)-7)];
PricePathShaper.w=[zeros(1,7),linspace(0, 1,ceil(T/2)),ones(1,T-ceil(T/2)-7)];
PricePathShaper.G=[zeros(1,7),ones(1,T-7)];
% The price path initial guess will be:
% PricePath0.price=p_eqm_initial.price + (p_eqm_final.price-p_eqm_initial.price)*PricePathShaper.price;
% Where after the first reveal, p_eqm_initial.price is understood to be the value on the path prior to the current reveal.
% Setting PricePathShaper.t0001.r and PricePathShaper.t0008.r we can have different initial guesses for the different reveals.
%
% It is possible for guessed shapes to differ by reveal.
% transpathoptions.usepreviouspathshapeasinitialguess=1;
% So the second reveal will use the solution to the first reveal as the
% shape for the initial guess for the price path. (If there was a third
% reveal it would use solution to the second reveal as the shape for the
% initial guess of the price path.) This is a good option when consecutive
% reveals are 'similar'.

% General eqm eqns, same idea as with the stationary general eqm
GeneralEqmEqns_Transition.capitalmarket=@(r,alpha,delta,K,L) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta); % r=marginal product of capital
GeneralEqmEqns_Transition.labormarket=@(w,alpha,K,L) w-(1-alpha)*(K^alpha)*(L^(-alpha)); % w=marginal product of labor
GeneralEqmEqns_Transition.govbudgetbalance=@(taxrevenue,G) taxrevenue-G;
% Note: in this example these are actually identical to the general eqm
% eqns for the stationary general eqm, but that is not often the case.

% Set up the shooting algorithm
transpathoptions.GEnewprice=3;
% Need to explain to transpathoptions how to use the GeneralEqmEqns to update the general eqm transition prices (in PricePath).
transpathoptions.GEnewprice3.howtoupdate=... % a row is: GEcondn, price, add, factor
    {'capitalmarket','r',0,0.3;...  % captialmarket GE condition will be positive if r is too big, so subtract
    'labormarket','w',0,0.3;... % labormarket GE condition will be positive if w is too big, so subtract
    'govbudgetbalance','G',1,0.5;... % govbudgetbalance GE condition will be negative if G is too big, so add
    };
% Note: the update is essentially new_price=price+factor*add*GEcondn_value-factor*(1-add)*GEcondn_value
% Notice that this adds factor*GEcondn_value when add=1 and subtracts it what add=0
% A small 'factor' will make the convergence to solution take longer, but too large a value will make it 
% unstable (fail to converge). Technically this is the damping factor in a shooting algorithm.

%% Solve the Multiple Reveal transition path
% Setup the options relating to the transition path
transpathoptions.verbose=1;
transpathoptions.maxiterations=200; % default is 1000
transpathoptions.fastOLG=1;
transpathoptions.graphpricepath=1; % plots of the ParamPath that get updated every interation
% transpathoptions.graphaggvarspath=1; % plots of the AggVarsPath that get updated every iteration

% Multiple reveal, solves both the final stationary eqm, and the transition path.
% It is possible to set vfoptions and simoptions seperately for each of
% these, but we will just use the same for both.
vfoptions_path=vfoptions;
simoptions_path=simoptions;
vfoptions_finaleqm=vfoptions;
simoptions_finaleqm=simoptions;
[RealizedPricePath, RealizedParamPath, PricePath, multirevealsummary]=MultipleRevealTransitionPath_Case1_FHorz(PricePathShaper, ParamPath, T, StationaryDist_init, jequaloneDist, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, GeneralEqmEqns_Transition, Params, DiscountFactorParamNames, AgeWeightParamNames, transpathoptions, simoptions_path, vfoptions_path, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm);
% Essentially, this is going, for each path, to first run 'HeteroAgentStationaryEqm_Case1_FHorz' to solve the final stationary
% general eqm for that path, and then will run 'TransitionPath_Case1_FHorz' to solve the path.

save tpath3B.mat -v7.3

%% With multiple reveal, most of the things you want are already in multirevealsummary
% for example
% multirevealsummary.RealizedVPath
% multirevealsummary.RealizedPolicyPath
% multirevealsummary.RealizedAgentDistPath
% multirevealsummary.RealizedAggVarsPath
% But we are going to compute them with separate commands anyway (just to see the commands)
% [In practice the only one of the following you are likely to need is the AggVars]

%% Now calculate some things about the transition path (path for Value fn, Policy fn, Agent Distribution)
% You can calculate the value and policy functions for the transition path
[RealizedVPath, RealizedPolicyPath, VPath, PolicyPath]=MultipleRevealValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath, T, Params, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions_path, vfoptions_finaleqm);
% Note: this includes calculating V_final and Policy_final based on
% PricePath in the final period of each reveal being the prices for the
% final stationary general eqm (for that reveal).

% You can then use these to calculate the agent distribution for the transition path
[RealizedAgentDistPath, AgentDistPath]=MultipleRevealAgentDistOnTransPath_Case1_FHorz(StationaryDist_init, jequaloneDist, PricePath, ParamPath, PolicyPath, AgeWeightParamNames,n_d,n_a,n_z,N_j,pi_z,T, Params, transpathoptions, simoptions);

%% Analyse the transition path
% And then we can calculate AggVars for the path
[RealizedAggVarsPath,AggVarsPath]=MultipleRevealEvalFnOnTransPath_AggVars_Case1_FHorz(FnsToEvaluate, AgentDistPath,PolicyPath, PricePath, ParamPath, Params, T, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, transpathoptions, simoptions);

%% Plot some paths
figure(1)
% Plot of K and r
% Note: include perios -3 to 0 (the initial stationary eqm) so can see any jump in period 1
% Solid line for realized path, with dotted lines for the two reveals so
% you can see how they combine to the final path
subplot(2,1,1); plot(1:1:T+7,RealizedAggVarsPath.K.Mean,'b-')
hold on
subplot(2,1,1); plot(1:1:T,AggVarsPath.t0001.K.Mean,'g.')
subplot(2,1,1); plot(8:1:T+7,AggVarsPath.t0008.K.Mean,'r.')
plot(-3:1:0,AllStats_init.K.Mean*ones(1,4),'r')
hold off
xlim([-3,T])
legend('Realised','reveal1','reveal2','Location','southeast')
title('Path of aggregate capital (K)')
subplot(2,1,2); plot(1:1:T+7,RealizedPricePath.r,'b-')
hold on
subplot(2,1,2); plot(1:1:T,PricePath.t0001.r,'g.')
subplot(2,1,2); plot(8:1:T+7,PricePath.t0008.r,'r.')
plot(-3:1:0,p_eqm_init.r*ones(1,4),'r')
hold off
xlim([-3,T])
title('Path of interest rate (r)')


