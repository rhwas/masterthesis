clear all; close all; clc;
PS3000Config;
%% Locate data
rootDirectory   = 'E:\Chalmers\TestProject\'; % Raw data location
directoryName   = 'w30'; % Directory to store processed data

%% Get raw data files for processing
fileList = dir([rootDirectory '*.mat']);
for y = 1:length(fileList)
    [filepath, name, ext] = fileparts(fileList(y).name);
    fileNames{y} = name;
end
% fileNames       = {'data-14-04-2020_13-03-33_empty_p90deg_nS-6000_nM-1000',...
%                    'data-14-04-2020_13-06-15_empty_m90deg_nS-1600_nM-250'};

%% Process each measurement
for z = 1:length(fileNames)
    Process(fileNames{z},rootDirectory, directoryName)
end