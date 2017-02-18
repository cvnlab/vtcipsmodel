%% Example script illustrating the Drift diffusion model (RT prediction) and various control models

%% Add code to the MATLAB path

% make sure to change this line to reflect where you have put
% the knkutils repository (http://github.com/kendrickkay/knkutils/)
addpath(genpath('/home/stone/kendrick/knkutils'));

%% Load data

% load in the data from the first experiment
a1 = load('experiment1.mat');

% prepare category labels
categories = a1.groupcategoryjudgment;
categories{1} = '';
categorytypes = {'WORD' 'FACE' 'OTHER'};

%% Prepare category vectors and projections

% which ROIs to extract data for?
whroi = [5 6 4];  % VWFA, FFA, hV4

% extract data
data = double(a1.groupbeta(whroi,:,:));  % ROIs x data points x tasks

% compute mean of each ROI during fixation task so that we can divide by it
datalen = mean(data(:,:,1),2);  % ROIs x 1

% based on fixation data, after normalization, compute centroids in 3D space
% and then normalize to be unit-length (producing three category vectors)
cvectors = [];  % 3 category vectors (word, face, other) x ROIs
for p=1:3
  ix = find(ismember(categories,categorytypes{p}));
  cvectors(p,:) = unitlength(mean(bsxfun(@rdivide,data(:,ix,1),datalen),2)');
end

% generate alternative category vectors, aligned with axes
avectors = eye(3);

% take each fixation response and project it onto its appropriate category vector.
% note that the blank stimulus falls out, so we have only 22 data points.
cprojection = [];  % 1 x data points
aprojection = [];  % 1 x data points
labelassts = [];   % 1 x data points (1=WORD, 2=FACE, 3=OTHER)
for p=1:size(data,2)
  if ~isempty(categories{p})
    thedata0 = data(:,p,1) ./ datalen;  % fixation response, 3 x 1
    whstim = find(ismember(categorytypes,categories{p}));  % which category is it?
    cprojection(end+1) = thedata0' * cvectors(whstim,:)';
    aprojection(end+1) = thedata0' * avectors(whstim,:)';
    labelassts(end+1)  = whstim;
  end
end

%% Prepare for model fitting

% define model names
modelnames = { ...
  'Flat' ...                                                   % Flat-response model that predicts the same response level for each data point
  'Drift diffusion model (separate thresholds)' ...            % Drift diffusion model with a separate threshold for each category
  'Drift diffusion model' ...                                  % Drift diffusion model
  'Drift diffusion model (axis-aligned category vectors)' ...  % Drift diffusion model using category vectors aligned with each axis of state space
  };

% calculate some things  
nr = 1;                   % number of ROIs we will be fitting (actually it's just 1 set of reaction times!)
nd = 22;                  % number of data points (22 reaction times)
nfolds = nd;              % number of folds of cross-validation
nm = length(modelnames);  % number of models

% prepare the data (categorization reaction times, ignoring blank stimulus)
data =   a1.grouprt(2:end)';    % 22 stimuli x 1
datase = a1.grouprtse(2:end)';  % 22 stimuli x 1

% compute noise ceiling:
%   nc is ROIs x 1
%   ncdist is ROIs x simulations
[nc,ncdist] = calcnoiseceiling(data',datase',[],[],@(x,y) calccorrelation(x,y,2));
%%

% define the metric to use when quantifying model accuracy.
% here we use Pearson's correlation.
metricfun = @(x,y) calccorrelation(x,y,1);

%% Fit models

% initialize outputs (details provided below)
modelfit =             NaN*zeros(nd,nr,nm);      % data points x ROIs x models
modelparams =          cell(1,nm);               % 1 x models (each element is parameters x ROIs)
modelpred =            NaN*zeros(nd,nr,nm);      % data points x ROIs x models
modelperformance =     NaN*zeros(nr,nm);         % ROIs x models

% fit models
for xx=1:2

  switch xx
  case 1

    % in this case, we do not cross-validate and instead just fit all the data
    xvalscheme = 0;
    extraopt = {'dosave','modelfit'};

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

    % Drift diffusion model (separate thresholds)
    case 2
      X = [cprojection' labelassts']; assert(size(X,1)==nd);
      seed0 = ones(1,4);
      opt1 = struct('stimulus',X,'data',data, ...
                    'model',{{[] [-Inf(1,4); Inf(1,4)] ...
                              @(p,x) p(1) + p(2)./x(:,1) .* (x(:,2)==1) + ...
                                            p(3)./x(:,1) .* (x(:,2)==2) + ...
                                            p(4)./x(:,1) .* (x(:,2)==3)}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % Drift diffusion model
    case 3
      X = [cprojection' labelassts']; assert(size(X,1)==nd);
      seed0 = ones(1,2);
      opt1 = struct('stimulus',X,'data',data, ...
                    'model',{{[] [-Inf(1,2); Inf(1,2)] ...
                              @(p,x) p(1) + p(2)./x(:,1)}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % Drift diffusion model (axis-aligned category vectors, separate thresholds)
    case 4
      X = [aprojection' labelassts']; assert(size(X,1)==nd);
      seed0 = ones(1,4);
      opt1 = struct('stimulus',X,'data',data, ...
                    'model',{{[] [-Inf(1,4); Inf(1,4)] ...
                              @(p,x) p(1) + p(2)./x(:,1) .* (x(:,2)==1) + ...
                                            p(3)./x(:,1) .* (x(:,2)==2) + ...
                                            p(4)./x(:,1) .* (x(:,2)==3)}}, ...
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
rr = 1;            % which ROI to look at (actually, it's just one set of reaction times)
whmodel = [3];     % which models to look at

% make a figure
figure; setfigurepos([100 100 500 200]); hold on;
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
ylabel('Reaction time (ms)');
legend(h2,modelnames(whmodel),'Location','EastOutside');
xlabel('Stimulus number');
title(sprintf('Modeling results for categorization reaction times'));
%%
