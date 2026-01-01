% Workshop OLG Transition Path 5

% Demographic transition
% ParamPath on conditional survival probabilities (sj) and age-masses
% (mewj). These are age-dependent parameters.

%% Setup for sj and mewj transitions
T=100;
N_j=81; % periods, represent ages 20 to 100% 
% 40 years of chaning demographics
% 60 years in final demographic state (to allow time to converge to final stationary general eqm)
% Conditional survival probabilities
Params.sj_init=[ones(1,46),linspace(1,0.99,81-46-10),linspace(0.99,0.9,9),0];
Params.sj_final=[ones(1,46),linspace(1,0.995,81-46-10),linspace(0.995,0.95,9),0]; % note: higher (so living longer)
ParamPath.sj=[Params.sj_init+(Params.sj_final-Params.sj_init).*linspace(0,1,40)'; Params.sj_final.*ones(60,1)];
% T-by-N_j (whether this or N_j-by_T, toolkit understands both)
% Calculate the implied mewj from the sj
ParamPath.mewj=ones(T,N_j);
for tt=1:T
    for jj=2:N_j
        ParamPath.mewj(tt,jj)=ParamPath.mewj(tt,jj-1)*ParamPath.sj(tt,jj-1); % mass of age jj is the mass of jj-1 that survive
    end
end
ParamPath.mewj=ParamPath.mewj./sum(ParamPath.mewj,2); % normalize age-masses to sum to one
% Looking at ParamPath.mewj you can see that as tt increases, the mass at older ages increases

% Note: This is rather incomplete, as really you should also have
% population growth rate n. But this does not change any thing in terms of
% the 'objects to compute'. Instead you need to renormalize the model for
% the population growth, and this just means you get a 'n' appearing ins
% some equations below. But other than 'n' in some equations, the way you
% do this with the toolkit does not change.

%% Model action and state-space
n_d=51; % number of grid points for our decision variable, labor supply
n_a=501; % number of grid points for our endogenous state, assets
n_z=9; % number of grid points for our exogenous markov state, labor productivity (per time worked; roughly hourly labor productivity)
% N_j=81; % periods, represent ages 20 to 100% 

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

% conditional survival probabilities (will be overwritten, just want it for setup)
Params.sj=Params.sj_init;

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
Params.mewj=ParamPath.mewj(1,:);
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
FnsToEvaluate.Beqreceived=@(h,aprime,a,z,sj,Beq,agej,Jr) Beq*(agej<Jr);
% First inputs are 'action space' (same as we did for ReturnFn), everything
% after this is interpreted as a parameter.

% Just make sure they are working okay
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);

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
Params.sj=ParamPath.sj(1,:);
Params.mewj=ParamPath.mewj(1,:);

[p_eqm_init,GEcondns_init]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
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
Params.sj=ParamPath.sj(T,:);
Params.mewj=ParamPath.mewj(T,:);

[p_eqm_final,GEcondns_final]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, [], pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
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
save tpath5A.mat
% load tpath5A.mat


%% Setup for the transition path
% T=100; % number of periods for transition path

% Already created ParamPath.sj and ParamPath.mewj

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

% Running, it was about stuck iterating around 2 or 3*10^(-4) but had clearly solved. So
transpathoptions.tolerance=4*10^(-4); % default is 10^(-4), which is a very demanding accuracy

% And go!
[PricePath,GECondnsPath]=TransitionPath_Case1_FHorz(PricePath0, ParamPath, T, V_final, AgentDist_init, jequaloneDist, n_d, n_a, n_z, N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, AgeWeightParamNames, transpathoptions, simoptions, vfoptions);

%% Now calculate some things about the transition path (path for Value fn, Policy fn, Agent Distribution)
% You can calculate the value and policy functions for the transition path
[VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath, T, V_final, Policy_final, Params, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);

% You can then use these to calculate the agent distribution for the transition path
AgentDistPath=AgentDistOnTransPath_Case1_FHorz(StationaryDist_init, jequaloneDist, PricePath, ParamPath, PolicyPath, AgeWeightParamNames,n_d,n_a,n_z,N_j,pi_z,T, Params, transpathoptions, simoptions);

%% Analyse the transition path
% And then we can calculate AggVars for the path
AggVarsPath=EvalFnOnTransPath_AggVars_Case1_FHorz(FnsToEvaluate, AgentDistPath,PolicyPath, PricePath, ParamPath, Params, T, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, transpathoptions, simoptions);

%%
save tpath5B.mat
% load tpath5B.mat

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

