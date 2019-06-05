function Example3D()
% EXAMPLE3D Provides a simple example on how to use Matlab3DThinning
%
%  USAGE: Example3D()
%
%
% __________________________________________________________________________
%  VARARGIN
%	See 'Parameter Initialization' section ...
%

% ------------------------------- Version 1.0 -------------------------------
%	Author:  Martin Wagner
%	Email:     mwagner9@wisc.edu
%	Created:  2019-06-05
% __________________________________________________________________________

%% Create Binary Volume
rng(1234);
tmp = zeros(512,512);
tmp(sub2ind(size(tmp), round(rand(5,1)*256+128), round(rand(5,1)*256+128))) = 1;

vol(:,:,512) = imdilate(tmp, strel('disk', 8, 0));
r = flip(cumsum(smooth(rand(1,511), 20)));
for k = 511:-1:1
    vol(:,:,k) = imrotate(vol(:,:,512), r(k), 'nearest', 'crop');
end

%% Perform Centerline Extraction
segs = Matlab3DThinning(vol, 0, 0); % Parameters 2 and 3 are optional to define pruning and smoothing.

%% Perform Centerline Extraction (With Pruning and Smoothing)
segs2 = Matlab3DThinning(vol, 5, 10); 

%% Display Results (All unsmoothed)
figure('Name', 'All centerlines unsmoothed');
for k = 1:length(segs)
    plot3(segs{k}(:,1)+1, segs{k}(:,2)+1, segs{k}(:,3)+1, 'LineWidth', 1); % One is added since coordinates are zero based
    hold on;
end
title('All Centerlines unsmoothed');
axis equal;

%% Display Results (Pruned and smoothed)
figure('Name', 'Pruned and smoothed centerlines');
for k = 1:length(segs2)
    plot3(segs2{k}(:,1)+1, segs2{k}(:,2)+1, segs2{k}(:,3)+1, 'LineWidth', 1); % One is added since coordinates are zero based
    hold on;
end
title('Pruned and smoothed centerlines');
axis equal;

