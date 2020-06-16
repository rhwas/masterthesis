clear all; close all; clc;
%% Locate Data
rootDirectory = 'E:\Chalmers\TestProject\'; % Raw data location
directoryName = 'w30'; % Directory to store input types

%% Get raw data filesnames
fileList = dir([rootDirectory '*.mat']);
for y = 1:length(fileList)
    [filepath, name, ext] = fileparts(fileList(y).name);
    fileNames{y} = name;
end
% fileNames = {'data-22-04-2020_14-06-05_triangle_nS-6000_nM-2000'};

%% Generate both input types
for j = 1:length(fileNames)
    clear('GenerateComplexBaseband')
    GenerateComplexBaseband(fileNames{j}, directoryName, rootDirectory)
    clear('GenerateSpectrogram')
    GenerateSpectrogram(fileNames{j}, directoryName, rootDirectory)
end

