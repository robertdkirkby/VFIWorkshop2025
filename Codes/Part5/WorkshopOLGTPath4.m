% Workshop OLG Transition Path 4

% Same reform as in Example 1
% Model now includes conditional survival probabilities and a general eqm
% eqn for bequests.

% Our reform:
tau_initial=0.1;
tau_final=0.05;

% NOTE: THIS IS NOT ACCURATE
% I have deliberately left this with not enough points on the asset grid so
% you can see what goes wrong and thus recognise how to deal with issues.
% Specifically, with the current n_a=351 the code is not accurate enough
% for the final stationart general eqm, you can see this looking at GEcondns_final
% where you can see that some are still non-zero in the third decimal
% place.
% You can still run the  transition path code, and it still 'solves', but
% if you look at the solution paths you can see a 'jump' in the final
% period that is of the same magnitude as the errors in the final genearal
% eqm (so roughly 10^-3). This happens because our initial guesses for
% PricePath0 all have the final stationary eqm values in the final period,
% and because VFI Toolkit never overwrites the period T of the price path
% while solving transition. This is delibrate as this way as long as you do
% use the final stationary eqm values for the last period of your initial
% guess for price path, you will always get this visible jump in your
% results if things are not working quite right.
% This can all be easily solved, just set n_a=501 and everything is
% accurate enough. I leave it here as an illustration of things to look for
% and how to deal with them.
% Actually, if you set n_a=501 on my laptop you get a 'Out of memory
% on device' gpu error [it cannot create such a big matrix on the gpu]. You
% can easily avoid this by setting transpathoptions.fastOLG=0, but of
% course then the code takes much longer to run. On my desktop however I
% have enough gpu memory and so don't get this error even with n_a=501.
% This also hints at another useful trick, namely you can code on your
% computer using small grids until you set everything up, then increase
% grids and send of to a more powerful gpu (say on a server you have access
% to).

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

% Bequests
Params.Beq=0.05; % initial guess for bequests (received)

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
DiscountFactorParamNames={'beta'};

% ReturnFn
ReturnFn=@(h,aprime,a,z,sigma,psi,eta,r,w,tau,kappa_j,Beq,agej,Jr)...
    WorkshopOLGTPath4_ReturnFn(h,aprime,a,z,sigma,psi,eta,r,w,tau,kappa_j,Beq,agej,Jr);
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
FnsToEvaluate.taxrevenue=@(h,aprime,a,z,w,tau,kappa_j) tau*w*kappa_j*h*exp(z); % effective labor supply
FnsToEvaluate.Beqleft=@(h,aprime,a,z,sj) aprime*(1-sj);
FnsToEvaluate.Beqreceived=@(h,aprime,a,z,sj,agej,Jr,Beq) Beq*(agej<Jr);
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Just make sure they are working okay
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,[],simoptions);

% For analysing the model
FnsToEvaluate2=FnsToEvaluate;
FnsToEvaluate2.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*h*exp(z); % w*kappa_j is the labor earnings
% Note: I keep the FnsToEvaluate use in general eqm to a minimum (to reduce
% runtimes) and then use FnsToEvaluate2 to analyse model with more stats.

%% General Eqm
GEPriceParamNames={'r','w','G','Beq'};
% note, Params.r we set earlier was an inital guess

GeneralEqmEqns.capitalmarket=@(r,alpha,delta,K,L) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta); % r=marginal product of capital
GeneralEqmEqns.labormarket=@(w,alpha,K,L) w-(1-alpha)*(K^alpha)*(L^(-alpha)); % w=marginal product of labor
GeneralEqmEqns.govbudgetbalance=@(taxrevenue,G) taxrevenue-G;
GeneralEqmEqns.bequests=@(Beqleft,Beqreceived) Beqleft-Beqreceived;
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
Params.Beq=p_eqm_init.Beq;

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

%% Solve for final stationary general eqm
Params.tau=tau_final; % final tax rate

[p_eqm_final,~,GEcondns_final]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% Done, the general eqm prices are in p_eqm
% GEcondns tells us the values of the GeneralEqmEqns, should be near zero

% To be able to analyze the general eqm, we need to use the r we found
Params.r=p_eqm_final.r;
Params.w=p_eqm_final.w;
Params.G=p_eqm_final.G;
Params.Beq=p_eqm_final.Beq;

% Evaluate the initial stationary general eqm
[V_final, Policy_final]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist_final=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy_final,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% Calculate various stats
AllStats_final=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist_final, Policy_final, FnsToEvaluate2, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions);
% Calculate the life-cycle profiles
AgeConditionalStats_final=LifeCycleProfiles_FHorz_Case1(StationaryDist_final,Policy_final, FnsToEvaluate2,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% Note: Only part of this final stationary general eqm we actually 'need'
% is the value fn (although we likely want p_eqm_final for initial guess of PricePath0). 
% Rest is just out of interest.

% Double-check that the general eqm is accurate before we start the
% transition path, because if it is not then it won't solve
GEcondns_final

%%
save tpath4A.mat
% load tpath4A.mat


%% Setup for the transition path
T=100; % number of periods for transition path

ParamPath.tau=tau_final*ones(1,T); % exogenous parameter path
% This is the reform that we want to model
% tau=0.05 is announced in period 1 (path is periods 1 to T)
% initial stationary dist corresponds to period 0
% But the agent dist from period 0 is the one that is relevant at start of period 1
% Note: all other parameters (except those in PricePath) are fixed/constant at the values in the Params structure.

% Initial guess for general eqm parameters
PricePath0.r=[linspace(p_eqm_init.r, p_eqm_final.r,ceil(T/2)), p_eqm_final.r*ones(1,T-ceil(T/2))];
PricePath0.w=[linspace(p_eqm_init.w, p_eqm_final.w,ceil(T/2)), p_eqm_final.w*ones(1,T-ceil(T/2))];
PricePath0.G=p_eqm_final.G*ones(1,T);
PricePath0.Beq=[linspace(p_eqm_init.Beq, p_eqm_final.Beq,ceil(T/2)), p_eqm_final.Beq*ones(1,T-ceil(T/2))];
% Just some reasonable guesses I made up.

% General eqm eqns, same idea as with the stationary general eqm
GeneralEqmEqns_Transition.capitalmarket=@(r,alpha,delta,K,L) r-(alpha*(K^(alpha-1))*(L^(1-alpha))-delta); % r=marginal product of capital
GeneralEqmEqns_Transition.labormarket=@(w,alpha,K,L) w-(1-alpha)*(K^alpha)*(L^(-alpha)); % w=marginal product of labor
GeneralEqmEqns_Transition.govbudgetbalance=@(taxrevenue,G) taxrevenue-G;
GeneralEqmEqns_Transition.bequests=@(Beqleft,Beqreceived) Beqleft_tminus1-Beqreceived; % Note: bequests are left in t-1 and received in t
% Note: in this example these are actually identical to the general eqm
% eqns for the stationary general eqm, but that is not often the case.

% Set up the shooting algorithm
transpathoptions.GEnewprice=3;
% Need to explain to transpathoptions how to use the GeneralEqmEqns to update the general eqm transition prices (in PricePath).
transpathoptions.GEnewprice3.howtoupdate=... % a row is: GEcondn, price, add, factor
    {'capitalmarket','r',0,0.3;...  % labormarket GE condition will be positive if r is too big, so subtract
    'labormarket','w',0,0.3;... % captialmarket GE condition will be positive if r is too big, so subtract
    'govbudgetbalance','G',1,0.5;... % govbudgetbalance GE condition will be negative if G is too big, so add
    'bequests','Beq',1,0.5;... % bequests GE condition will be negative if Beq is too big, so add
    };
% Note: the update is essentially new_price=price+factor*add*GEcondn_value-factor*(1-add)*GEcondn_value
% Notice that this adds factor*GEcondn_value when add=1 and subtracts it what add=0
% A small 'factor' will make the convergence to solution take longer, but too large a value will make it 
% unstable (fail to converge). Technically this is the damping factor in a shooting algorithm.

%% Solve the transition path
% Setup the options relating to the transition path
transpathoptions.verbose=1;
transpathoptions.maxiterations=200; % default is 1000
transpathoptions.fastOLG=1;
transpathoptions.graphpricepath=1; % plots of the ParamPath that get updated every interation
% transpathoptions.graphaggvarspath=1; % plots of the AggVarsPath that get updated every iteration
% And go!
PricePath=TransitionPath_Case1_FHorz(PricePath0, ParamPath, T, V_final, AgentDist_init, jequaloneDist, n_d, n_a, n_z, N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, AgeWeightParamNames, transpathoptions, simoptions, vfoptions);

%% Now calculate some things about the transition path (path for Value fn, Policy fn, Agent Distribution)
% You can calculate the value and policy functions for the transition path
[VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath, T, V_final, Policy_final, Params, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);

% You can then use these to calculate the agent distribution for the transition path
AgentDistPath=AgentDistOnTransPath_Case1_FHorz(StationaryDist_init, jequaloneDist, PricePath, ParamPath, PolicyPath, AgeWeightParamNames,n_d,n_a,n_z,N_j,pi_z,T, Params, transpathoptions, simoptions);

%% Analyse the transition path
% And then we can calculate AggVars for the path
AggVarsPath=EvalFnOnTransPath_AggVars_Case1_FHorz(FnsToEvaluate, AgentDistPath,PolicyPath, PricePath, ParamPath, Params, T, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, transpathoptions, simoptions);

%%
save tpath4B.mat
% load tpath4B.mat

%% Plot some paths
figure(1)
% Plot of K and r
% Note: include perios -3 to 0 (the initial stationary eqm) so can see any jump in period 1
subplot(2,1,1); plot(1:1:T,AggVarsPath.K.Mean)
hold on
plot(-3:1:0,AllStats_init.K.Mean*ones(1,4),'r')
hold off
xlim([-3,T])
title('Path of aggregate capital (K)')
subplot(2,1,2); plot(1:1:T,PricePath.r)
hold on
plot(-3:1:0,p_eqm_init.r*ones(1,4),'r')
hold off
xlim([-3,T])
title('Path of interest rate (r)')

