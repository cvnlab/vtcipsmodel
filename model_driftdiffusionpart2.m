%% Example script illustrating the Drift diffusion model (IPS prediction) 

%% Add code to the MATLAB path

% make sure to change this line to reflect where you have put
% the knkutils repository (http://github.com/kendrickkay/knkutils/)
addpath(genpath('/home/stone/kendrick/knkutils'));

%% Load data

% load in the data from the first experiment
a1 = load('experiment1.mat');

%% Prepare for model fitting

% define model names
modelnames = { ...
  'Flat' ...             % Flat-response model that predicts the same response level for each data point
  'RT monotonic' ...     % Monotonic function of reaction time
  };
  
% which ROIs do we want to fit?
whroi = [8];  % IPS

% calculate some things
nr = length(whroi);       % number of ROIs we will be fitting
nd = 22;                  % number of data points (categorization task, exclude blank stimulus)
nfolds = nd;              % number of folds of cross-validation 
nm = length(modelnames);  % number of models

% prepare the data (group-averaged beta weights during categorization task, exclude blank stimulus)
data =        squish(permute(double(a1.groupbeta(whroi,2:end,2)),[2 3 1]),2);     % 22 conditions x ROIs
datase =      squish(permute(double(a1.groupbetase(whroi,2:end,2)),[2 3 1]),2);   % 22 conditions x ROIs

% compute noise ceiling:
%   nc is ROIs x 1
%   ncdist is ROIs x simulations
[nc,ncdist] = calcnoiseceiling(data',datase');
%%

% define the metric to use when quantifying model accuracy.
% we use an R^2 metric where variance is computed relative to 0% BOLD change.
metricfun = @(x,y) calccod(x,y,1,0,0);

%% Fit models

% initialize outputs (details provided below)
modelfit =             NaN*zeros(nd,nr,nm);      % data points x ROIs x models
modelparams =          cell(1,nm);               % 1 x models (each element is parameters x ROIs)
modelpred =            NaN*zeros(nd,nr,nm);      % data points (2*n) x ROIs x models
modelperformance =     NaN*zeros(nr,nm);         % ROIs x models

% fit models
for xx=1:2

  switch xx
  case 1

    % in this case, we do not cross-validate and instead just fit all the data
    xvalscheme = 0;
    extraopt = {'dosave','modelfit'};  % indicate that we want the 'modelfit' output

  case 2

    % in this case, we perform cross-validation, so we need to define the cross-validation scheme
    xvalscheme = ones(nfolds,nd);
    for p=1:nfolds
      ix = picksubset(1:nd,[nfolds p]);
      xvalscheme(p,ix) = -1;
    end
    extraopt = {};

    % compute how we can go back to the original order
    [d,xvalschemeREV] = resamplingtransform(xvalscheme);

  end

  % loop over models
  for mm=1:nm

    switch mm

    % Flat-response model
    case 1
      X = ones(nd,1);
      seed0 = 0.1 * ones(1,1);
      opt1 = struct('stimulus',X,'data',data, ...
                    'model',{{[] [-Inf(1,1); Inf(1,1)] @(p,x) x*p'}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % RT monotonic model
    case 2
      X = calczscore(a1.grouprt(2:end)'); assert(size(X,1)==nd);
      seed0 = [.5 1 0 .5];
      opt1 = struct('stimulus',X,'data',data, ...
                    'model',{{[] [-Inf(1,4); Inf(1,4)] ...
                              @(p,x) p(1)*tanh(p(2)*x+p(3))+p(4)}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    end

    % finally, fit the model
    results = fitnonlinearmodel(opt1);

    % take the results and store them
    switch xx
    case 1
      modelfit(:,:,mm)       = squish(results.modelfit(1,:,:),2);
      modelparams{mm}        = squish(results.params(1,:,:),2);
    case 2
      modelpred(:,:,mm)      = results.modelpred;
      modelperformance(:,mm) = results.aggregatedtestperformance(1,:);
    end

  end

end
%%

% undo the effect of the cross-validation re-ordering. after this step,
% the data points are back in the original order.
modelpred = modelpred(xvalschemeREV,:,:);

% ok, the model fitting is complete.
%
% modeling results are compiled into the following variables:
% - modelfit is data points x ROIs x models. this gives, for each model
%   applied to each ROI, the model fit to all data points (no cross-validation).
% - modelparams is a cell vector that is 1 x models. each element is parameters x ROIs,
%   which stores the estimated parameters from each model applied to each ROI.
% - modelpred is data points x ROIs x models. this is the set of cross-validated 
%   model predictions, aggregated across all cross-validation iterations.
% - modelperformance is ROIs x models. this is the quantification of model
%   cross-validation accuracy.

%% Inspect modeling results

% define
rr = 1;            % which ROI to look at
whmodel = [2];     % which model to look at

% make a figure
figure; setfigurepos([100 100 600 200]); hold on;
xxx = 1+(1:nd);  % leave a spot for the blank stimulus
yyy =   data(:,rr);
yyyse = datase(:,rr);
h = bar(xxx,yyy,1);
set(h,'FaceColor','k');
set(errorbar2(xxx,yyy,yyyse,'v','k-','LineWidth',2),'Color',[.5 .5 .5]);
cmap0 = [1 0 0];
h = []; h2 = [];
for mm=1:length(whmodel)
  h(mm)  = plot(xxx, modelfit(:,rr,whmodel(mm)),'o-','Color',(cmap0(mm,:)+2*[1 1 1])/3,'LineWidth',2);
  h2(mm) = plot(xxx,modelpred(:,rr,whmodel(mm)),'o-','Color',cmap0(mm,:),'LineWidth',2);
end
ylabel('BOLD response (% change)');
legend(h2,modelnames(whmodel),'Location','EastOutside');
xlabel('Stimulus number');
title(sprintf('Modeling results for %s',a1.roilabels{whroi(rr)}));
%%
