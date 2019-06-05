function Example2D()
% EXAMPLE2D Provides a simple example on how to use Matlab2DThinning
%
%  USAGE: Example2D()
%
%

% ------------------------------- Version 1.0 -------------------------------
%	Author:  Martin Wagner
%	Email:     mwagner9@wisc.edu
%	Created:  2019-06-05
% __________________________________________________________________________

%% Create Binary Image
img = imread('cameraman.tif');
[gx, gy] = gradient(imgaussfilt(single(img), 2.0));
g = sqrt(gx.^2 + gy.^2) > 12;

%% Perform Centerline Extraction
segs = Matlab2DThinning(g);

%% Perform Centerline Extraction with Tree Pruning
segs_pruned = Matlab2DThinning(g, 10);

%% Perform Centerline Extraction with Tree Pruning and Smoothing
segs_smoothed = Matlab2DThinning(g, 5, 10);

%% Display Results (All unsmoothed)
c = jet(length(segs));
figure('Name', 'All Centerlines unsmoothed');
imshow(g);
hold on;
for k = 1:length(segs)
    plot(segs{k}(:,1)+1, segs{k}(:,2)+1, 'Color', c(k,:)); % One is added since coordinates are zero based
end
title('All Centerlines unsmoothed');

%% Display Results (Pruned)
c = jet(length(segs_pruned));
figure('Name', 'Pruned Centerlines unsmoothed');
imshow(g);
hold on;
for k = 1:length(segs_pruned)
    plot(segs_pruned{k}(:,1)+1, segs_pruned{k}(:,2)+1, 'Color', c(k,:)); % One is added since coordinates are zero based
end
title('Pruned Centerlines unsmoothed');

%% Display Results (Pruned)
c = jet(length(segs_smoothed));
figure('Name', 'Pruned Centerlines smoothed');
imshow(g);
hold on;
for k = 1:length(segs_smoothed)
    plot(segs_smoothed{k}(:,1)+1, segs_smoothed{k}(:,2)+1, 'Color', c(k,:)); % One is added since coordinates are zero based
end
title('Pruned Centerlines smoothed');
