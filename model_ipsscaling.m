%% Example script illustrating the IPS-scaling model and various control models

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
  'Flat' ...                     % Flat-response model that predicts the same response level for each data point
  'Task-invariant' ...           % Task has no effect (use one set of responses to fit all tasks)
  'Additive' ...                 % Add a constant (same for both tasks)
  'AdditiveTS' ...               % Add a separate constant for each task
  'Scaling' ...                  % Scale by a constant (same for both tasks)
  'ScalingTS' ...                % Scale by a separate constant for each task
  'AreaSpecificWord' ...         % Like ScalingTS but the scaling occurs only for words
  'AreaSpecificFace' ...         % Like ScalingTS but the scaling occurs only for faces
  'IPS-additive' ...             % Add a scaled version of the IPS signal
  'IPS-scaling' ...              % Multiply by a scaled version of the IPS signal
  };

% which ROIs do we want to fit?
whroi = [5 6];  % VWFA, FFA

% which ROI supplies the top-down signal?
whtopdown = 8;  % IPS

% calculate some things
n = 23;                   % number of stimuli
nr = length(whroi);       % number of ROIs we will be fitting
nd = 3*n;                 % number of data points (3 tasks, 23 stimuli)
nfolds = 2*n;             % number of folds of cross-validation (we resample over the categorization and one-back tasks)
nm = length(modelnames);  % number of models

% prepare the data (group-averaged beta weights during all three tasks)
data =        squish(permute(double(a1.groupbeta(whroi,:,:)),[2 3 1]),2);     % 23*3 conditions x ROIs
datase =      squish(permute(double(a1.groupbetase(whroi,:,:)),[2 3 1]),2);   % 23*3 conditions x ROIs
datatopdown = squish(permute(double(a1.groupbeta(whtopdown,:,:)),[2 3 1]),2); % 23*3 conditions x 1

% repeat datatopdown for code convenience
datatopdown = repmat(datatopdown,[1 size(data,2)]);                           % 23*3 conditions x ROIs

% insert NaNs into datatopdown for the fixation responses, so that 
% datatopdown does not influence the model for these data points.
% this is handled in the model fitting below.
datatopdown(1:n,:) = NaN;                                                     % 23*3 conditions x ROIs

% NOTE:
% - A few control models are evaluated in the paper but are not explicitly done here
%   in order to keep the code compact.
%   - To implement "IPS-scaling (shuffle)", one would perform:
%       datatopdown = cat(1,datatopdown(1:n,:), ...
%                           permutedim(datatopdown(n+(1:2*n),:),1,[],1));
%   - To implement "IPS-scaling (shuffle within task)", one would perform:
%       datatopdown = cat(1,datatopdown(1:n,:), ...
%                           permutedim(datatopdown(n+(1:n),:),1,[],1), ...
%                           permutedim(datatopdown(2*n+(1:n),:),1,[],1));

% compute noise ceiling:
%   nc is ROIs x 1
%   ncdist is ROIs x simulations
[nc,ncdist] = calcnoiseceiling(data(n+(1:2*n),:)',datase(n+(1:2*n),:)');
%%

% define the metric to use when quantifying model accuracy.
% we use an R^2 metric where variance is computed relative to 0% BOLD change.
metricfun = @(x,y) calccod(x,y,1,0,0);

% prepare category labels
categories = a1.groupcategoryjudgment;
categories{1} = '';

%% Fit models

% initialize outputs (details provided below)
modelfit =             NaN*zeros(nd,nr,nm);      % data points x ROIs x models
modelparams =          cell(1,nm);               % 1 x models (each element is parameters x ROIs)
modelpred =            NaN*zeros(2*n,nr,nm);     % data points (2*n) x ROIs x models
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
      ix = picksubset(1:2*n,[nfolds p]);
      xvalscheme(p,n+ix) = -1;  % notice that the cross-validation is done over the categorization and one-back tasks
    end
    extraopt = {};
    
    % compute how we can go back to the original order
    [d,xvalschemeREV] = resamplingtransform(xvalscheme(:,n+(1:2*n)));

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

    % Task-invariant model
    case 2
      X = repmat(eye(n),[3 1]);
      seed0 = @(ix) data(1:n,ix)';
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n); Inf(1,n)] @(p,x) x*p'} ...
                             {@(ss) ss [-Inf(1,n); Inf(1,n)] @(ss) @(p,x) x*p'}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % Additive model
    case 3
      X = repmat([eye(n) zeros(n,1)],[3 1]);
      X(1:n,n+1) = 1;
      X(n+(1:2*n),n+1) = 2;
      seed0 = @(ix) [data(1:n,ix)' 0 0];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN -Inf; Inf(1,n+2)] @(p,x) x(:,1:n)*p(1:n)' + p(n+x(:,n+1))'} ...
                             {@(ss) ss [-Inf(1,n) NaN -Inf; Inf(1,n+2)] ...
                                                           @(ss) @(p,x) x(:,1:n)*p(1:n)' + p(n+x(:,n+1))'}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % AdditiveTS model
    case 4
      X = repmat([eye(n) zeros(n,1)],[3 1]);
      X(1:n,n+1) = 1;
      X(n+(1:n),n+1) = 2;
      X(2*n+(1:n),n+1) = 3;
      seed0 = @(ix) [data(1:n,ix)' 0 0 0];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN -Inf -Inf; Inf(1,n+3)] @(p,x) x(:,1:n)*p(1:n)' + p(n+x(:,n+1))'} ...
                             {@(ss) ss [-Inf(1,n) NaN -Inf -Inf; Inf(1,n+3)] ...
                                                                @(ss) @(p,x) x(:,1:n)*p(1:n)' + p(n+x(:,n+1))'}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % Scaling model
    case 5
      X = repmat([eye(n) zeros(n,1)],[3 1]);
      X(1:n,n+1) = 1;
      X(n+(1:2*n),n+1) = 2;
      seed0 = @(ix) [data(1:n,ix)' 1 1];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN -Inf; Inf(1,n+2)] @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'} ...
                             {@(ss) ss [-Inf(1,n) NaN -Inf; Inf(1,n+2)] ...
                                                           @(ss) @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % ScalingTS model
    case 6
      X = repmat([eye(n) zeros(n,1)],[3 1]);
      X(1:n,n+1) = 1;
      X(n+(1:n),n+1) = 2;
      X(2*n+(1:n),n+1) = 3;
      seed0 = @(ix) [data(1:n,ix)' 1 1 1];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN -Inf -Inf; Inf(1,n+3)] @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'} ...
                             {@(ss) ss [-Inf(1,n) NaN -Inf -Inf; Inf(1,n+3)] ...
                                                                @(ss) @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});
    
    % AreaSpecificWord model
    case 7
      X = repmat([eye(n) zeros(n,1)],[3 1]);
      specialix = find(ismember(categories,'WORD'));
      X(:,n+1) = 1;               % default is 1
      X(n+specialix,n+1) = 2;     % words are allowed to change in categorization task
      X(2*n+specialix,n+1) = 3;   % words are allowed to change in one-back task
      seed0 = @(ix) [data(1:n,ix)' 1 1 1];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN -Inf -Inf; Inf(1,n+3)] @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'} ...
                             {@(ss) ss [-Inf(1,n) NaN -Inf -Inf; Inf(1,n+3)] ...
                                                           @(ss) @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % AreaSpecificFace model
    case 8
      X = repmat([eye(n) zeros(n,1)],[3 1]);
      specialix = find(ismember(categories,'FACE'));
      X(:,n+1) = 1;                % default is 1
      X(n+specialix,n+1) = 2;      % faces are allowed to change in categorization task
      X(2*n+specialix,n+1) = 3;    % faces are allowed to change in one-back task
      seed0 = @(ix) [data(1:n,ix)' 1 1 1];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN -Inf -Inf; Inf(1,n+3)] @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'} ...
                             {@(ss) ss [-Inf(1,n) NaN -Inf -Inf; Inf(1,n+3)] ...
                                                           @(ss) @(p,x) x(:,1:n)*p(1:n)' .* p(n+x(:,n+1))'}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % IPS-additive model
    case 9
      X = [repmat(eye(n),[3 1]) (1:3*n)'];
      seed0 = @(ix) [data(1:n,ix)' datatopdown(:,ix)' 0 0];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN(1,3*n) -Inf -Inf; Inf(1,n+3*n+2)] ...
                      @(p,x) x(:,1:n)*p(1:n)' + nanreplace(p(n+3*n+1)*p(n+x(:,n+1))'+p(n+3*n+2))} ...
                             {@(ss) ss [-Inf(1,n) NaN(1,3*n) -Inf -Inf; Inf(1,n+3*n+2)] ...
                @(ss) @(p,x) x(:,1:n)*p(1:n)' + nanreplace(p(n+3*n+1)*p(n+x(:,n+1))'+p(n+3*n+2))}}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});
    
    % IPS-scaling model
    case 10
      X = [repmat(eye(n),[3 1]) (1:3*n)'];
      seed0 = @(ix) [data(1:n,ix)' datatopdown(:,ix)' 0 1];
      opt1 = struct('stimulus',X,'data',@(ix) data(:,ix),'vxs',1:size(data,2), ...
                    'model',{ ...
                            {{[] [NaN(1,n) NaN(1,3*n) -Inf -Inf; Inf(1,n+3*n+2)] ...
                      @(p,x) x(:,1:n)*p(1:n)' .* nanreplace(p(n+3*n+1)*p(n+x(:,n+1))'+p(n+3*n+2),1)} ...
                             {@(ss) ss [-Inf(1,n) NaN(1,3*n) -Inf -Inf; Inf(1,n+3*n+2)] ...
                @(ss) @(p,x) x(:,1:n)*p(1:n)' .* nanreplace(p(n+3*n+1)*p(n+x(:,n+1))'+p(n+3*n+2),1)}}}, ...
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
% the data points are back in the original order (across the categorization
% and one-back tasks).
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
whmodel = [2 10];  % which models to look at

% make a figure
figure; setfigurepos([100 100 950 250]); hold on;
xxx = 1:3*n;
xxxALT = n+(1:2*n);
yyy =   data(:,rr);
yyyse = datase(:,rr);
h = bar(xxx,yyy,1);
set(h,'FaceColor','k');
set(errorbar2(xxx,yyy,yyyse,'v','k-','LineWidth',2),'Color',[.5 .5 .5]);
cmap0 = [0 0 1;
         1 0 0];
h = []; h2 = [];
for mm=1:length(whmodel)
  h(mm)  = plot(xxx,    modelfit(:,rr,whmodel(mm)),'o-','Color',(cmap0(mm,:)+2*[1 1 1])/3,'LineWidth',2);
  h2(mm) = plot(xxxALT,modelpred(:,rr,whmodel(mm)),'o-','Color',cmap0(mm,:),'LineWidth',2);
end
ylabel('BOLD response (% change)');
legend(h2,modelnames(whmodel),'Location','EastOutside');
xlabel('Stimulus number');
title(sprintf('Modeling results for %s',a1.roilabels{whroi(rr)}));
%%
