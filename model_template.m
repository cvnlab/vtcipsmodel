%% Example script illustrating the Template model and various control models

%% Add code to the MATLAB path

% make sure to change this line to reflect where you have put
% the knkutils repository (http://github.com/kendrickkay/knkutils/)
addpath(genpath('/home/stone/kendrick/knkutils'));

%% Load data

% load in the data from the first experiment
a1 = load('experiment1.mat');

%% Do some inspections

% look at the stimuli (first frame of each of 23 stimuli).
% the order is down and then right.
figure; setfigurepos([100 100 600 400]);
imagesc(makeimagestack(permute(double(a1.stimuli(:,:,1,:)),[1 2 4 3]),[],0));
colormap(gray); colorbar;
axis image tight;
title('Stimuli');
%%

% look at the group-averaged beta weights for VWFA during the fixation task
figure; setfigurepos([100 100 600 200]); hold on;
beta =   a1.groupbeta(5,:,1);
betase = a1.groupbetase(5,:,1);
bar(beta);
errorbar2(1:length(beta),beta,betase,'v','r-','LineWidth',2);
xlabel('Stimulus number');
ylabel('BOLD signal (% change)');
title('VWFA responses during fixation task');
%%

%% Perform some pre-processing of the stimuli

% The rationale is that there are a series of image processing steps
% that occur before the free parameters of the model take influence.
% Thus, we can do these computations up front before we move
% on to model fitting.

% convert to single format so that we can actually do computations.
% also, reshape so that all of the stimulus images lie along the third dimension.
X = reshape(single(a1.stimuli),[500 500 10*23]);

% resize the images to 250 pixels x 250 pixels to reduce computational burden.
X = single(processmulti(@imresize,double(X),[250 250]));  % dimensions are now 250 x 250 x 10*23

% note that pixel values are proportional to actual luminance values,
% so no special handling of gamma is required.

% convert the pixel range from [0,254] to [-.5,.5] by rescaling values
% and subtracting off the value corresponding to the gray background.
% after this step, the gray background is at 0.
X = (X/254) - .5;

%% Prepare V1-like representation of the stimuli

% define
pxtodeg = 1/240 * 2;       % 240 pixels correspond to 2 degrees of visual angle
stimsize = 500 * pxtodeg;  % compute the total size of the stimulus in deg

% next, we want to project onto V1-like filters.
% we will use filters tuned at 4 cycles per deg.

% how many cycles per stimulus field-of-view should the filters be tuned at?
cpfov = 4*stimsize;

% go ahead and do the projection.  we will get a lot of outputs, but 
% the one we care about is f, which has dimensions 10*23 images x 16384 channels.
% the channels are ordered like (2 phases)*(8 orientations)*(32 positions x 32 positions).
[f,gbrs,gaus,sds,indices,info] = applymultiscalegaborfilters(squish(X,2)',cpfov,-1,1,8,2,0.01,2,0);
%%

% combine quadrature-phase filter outputs (complex-cell energy model)
f = sqrt(blob(f.^2,2,2));  % dimensions are now 10*23 images x 8192 channels

% inspect filter outputs for the first image of the 5th stimulus.
% the order is 0 deg (horizontal filter), 22.5 deg, 44.5 deg, etc., rotating counter-clockwise.
figure; setfigurepos([100 100 800 200]);
imagesc(makeimagestack(permute(reshape(f(10*4+1,:),8,32,32),[2 3 1]),[],[],[1 8]));
colormap(hot); colorbar;
axis image tight;
title('Filter outputs');
%%

% now we want to apply divisive normalization.

% do some reshaping
f = permute(reshape(f,10,23,[]),[1 3 2]);  % now: 10 x 8192 channels x 23

% compute the average across 8 orientations (at each position) and place in the imaginary component
f = f + j*upsamplematrix(blob(f,2,8)/8,8,2,[],'nearest');

% define parameters (taken identically from Kay et al., PLOS Comp Bio, 2013)
ee = 1;
ss = 0.5;

% perform divisive normalization
f = real(f).^ee./(ss.^ee+imag(f).^ee);

% the dimensions of f are now 10 images x 8192 channels x 23 stimuli.

% NOTE:
% - In the paper, one of the control models is called "Template model (omit first stage)".
%   This model omits the first stage of the model and computes a template operation on a 
%   pixel representation of the stimuli. To implement this model, instead of performing 
%   the steps above, one would just prepare f as:
%     f = permute(reshape(squish(X+.5,2),250*250,10,23),[2 1 3]);
%   This sets up f with dimensions 10 images x (250 pixels * 250 pixels) x 23 stimuli
%   and places the units of f into the range [0,1].

%% Compute templates

% which stimuli do we want to create templates from?
templateix = [9 5];   % word, face

% calculate templates
templates = [];  % dimensions will be 2 templates x 8192 channels
for qq=1:length(templateix)
  templates(qq,:) = mean(f(:,:,templateix(qq)),1);  % compute the mean of the 10 images in the V1-like representation
end

% NOTE:
% - A few different control models are evaluated in the paper, based on modifying
%   the template used in the model.
%   - To implement "Template model (non-selective template)", one would do:
%       templates(3,:) = 1;
%   - To implement "Template model (mixed template)", one would do:
%       templates(3,:) = mean(unitlength(templates(1:2,:),2),1);  % this template is a mix of "word" and "face"
%   - To implement "Template model (random template)", one would do:
%       templates(3,:) = rand(1,size(templates,2));

%% Pre-compute some quantities

% To reduce computational time, we pre-compute as much as we can before we proceed to model fitting.

% compute "A" which is the dot-product of the stimulus and the template (S-dot-T in the paper)
A = squish(permute(f,[1 3 2]),2) * templates';  % 10*23 images x 2 templates

% compute "B" which is the average of all of the channels in the stimulus (S-bar in the paper)
B = mean(squish(permute(f,[1 3 2]),2),2);       % 10*23 images x 1

%% Prepare for model fitting

% define model names
modelnames = { ...
  'Flat' ...                % Flat-response model that predicts the same response level for each data point 
  'CategoryWord' ...        % Category model using words vs. non-words
  'CategoryFace' ...        % Category model using faces vs. non-faces
  'TemplateWordOnlySub' ... % Template model (only subtractive normalization) using a word template
  'TemplateWordOnlyDiv' ... % Template model (only divisive normalization) using a word template
  'TemplateWord' ...        % Template model using a word template
  'TemplateFaceOnlySub' ... % Template model (only subtractive normalization) using a face template
  'TemplateFaceOnlyDiv' ... % Template model (only divisive normalization) using a face template
  'TemplateFace' ...        % Template model using a face template
  };

% which ROIs do we want to fit?
whroi = [5 6];  % VWFA, FFA

% calculate some things
nr = length(whroi);      % number of ROIs we will be fitting
nd = 23;                 % number of data points
nfolds = 23;             % number of folds of cross-validation to perform
nm = length(modelnames); % number of models

% prepare the data (group-averaged beta weights during the fixation task)
data =   permute(double(a1.groupbeta(whroi,:,1)),[2 1 3]);    % 23 stimuli x ROIs
datase = permute(double(a1.groupbetase(whroi,:,1)),[2 1 3]);  % 23 stimuli x ROIs

% compute noise ceilings:
%   nc is ROIs x 1
%   ncdist is ROIs x simulations
[nc,ncdist] = calcnoiseceiling(data',datase');
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
modelpred =            NaN*zeros(nd,nr,nm);      % data points x ROIs x models
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

    % Category model using words vs. non-words
    case 2
      ix = ismember(categories,'WORD');
      X = zeros(nd,2);
      X( ix,1) = 1;  % first column is 1 for words
      X(~ix,2) = 1;  % second column is 1 for non-words
      seed0 = 0.1 * ones(1,2);
      opt1 = struct('stimulus',X,'data',data, ...
                    'model',{{[] [-Inf(1,2); Inf(1,2)] @(p,x) x*p'}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});

    % Category model using faces vs. non-faces
    case 3
      ix = ismember(categories,'FACE');
      X = zeros(nd,2);
      X( ix,1) = 1;  % first column is 1 for faces
      X(~ix,2) = 1;  % second column is 1 for non-faces
      seed0 = 0.1 * ones(1,2);
      opt1 = struct('stimulus',X,'data',data, ...
                    'model',{{[] [-Inf(1,2); Inf(1,2)] @(p,x) x*p'}}, ...
                    'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                    'optimoptions',{{'Display','off'}},extraopt{:});
    
    % Template models
    case {4 5 6 7 8 9}
    
      % calculate some flags
      complextyp = mod2(mm,3);       % 1 means only subtractive; 2 means only divisive; 3 means both
      templatenum = ceil((mm-3)/3);  % 1 means use word template; 2 means use face template

      % pre-condition A and B. we collect the results into
      % stimAB which is 10*23 images x 2, where the columns contain the
      % pre-conditioned A and the pre-conditioned B.
      stimAB = [A(:,templatenum)/mean(A(:,templatenum)) B/mean(B)];
      
      switch complextyp
      
      % this is the only subtractive model
      case 1
        cs = [0 .5 1 1.5 2 3 5];
        seed0 = [];
        cnt = 1;
        for q=1:length(cs)
          seed0(cnt,:) = [10 cs(q)];
          cnt = cnt + 1;
        end
        opt1 = struct('stimulus',permute(reshape(stimAB,[10 nd 2]),[2 3 1]),'data',data, ...
                      'model',{{[] [-Inf(1,2); Inf(1,2)] ...
                                @(p,x) p(1) * posrect(x(:,1)-p(2)*x(:,2))}}, ...
                      'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                      'optimoptions',{{'Display','off'}},extraopt{:});
      
      % this is the only divisive model
      case 2
        bs = [.01 .05 .1 .5 1 5 10];
        seed0 = [];
        cnt = 1;
        for p=1:length(bs)
          seed0(cnt,:) = [10 bs(p)];
          cnt = cnt + 1;
        end
        opt1 = struct('stimulus',permute(reshape(stimAB,[10 nd 2]),[2 3 1]),'data',data, ...
                      'model',{{[] [-Inf(1,2); Inf(1,2)] ...
                                @(p,x) p(1) * (x(:,1) ./ (p(2)+x(:,2)))}}, ...
                      'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                      'optimoptions',{{'Display','off'}},extraopt{:});

      % this is the full model
      case 3
        bs = [.01 .05 .1 .5 1 5 10];
        cs = [0 .5 1 1.5 2 3 5];
        seed0 = [];
        cnt = 1;
        for p=1:length(bs)
          for q=1:length(cs)
            seed0(cnt,:) = [10 bs(p) cs(q)];
            cnt = cnt + 1;
          end
        end
        opt1 = struct('stimulus',permute(reshape(stimAB,[10 nd 2]),[2 3 1]),'data',data, ...
                      'model',{{[] [-Inf(1,3); Inf(1,3)] ...
                                @(p,x) p(1) * (posrect(x(:,1)-p(3)*x(:,2)) ./ (p(2)+x(:,2)))}}, ...
                      'seed',seed0,'resampling',xvalscheme,'metric',metricfun, ...
                      'optimoptions',{{'Display','off'}},extraopt{:});
      end

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
rr = 1;           % which ROI to look at
whmodel = [2 6];  % which models to look at

% make a figure
figure; setfigurepos([100 100 600 250]); hold on;
xxx = 1:nd;
yyy =   data(:,rr);
yyyse = datase(:,rr);
h = bar(xxx,yyy,1);
set(h,'FaceColor','k');
set(errorbar2(xxx,yyy,yyyse,'v','k-','LineWidth',2),'Color',[.5 .5 .5]);
cmap0 = [0 0 1;
         1 0 0];
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
