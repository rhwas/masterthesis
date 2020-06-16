clear all; close all; clc;
%% Load Data
rootDirectory               = 'E:\Chalmers\TestProject\';
destinationDirectoryName    = 'w30';
%% List of objects and the path to the prepared data
fileNames_complex           = {'combinedData_complexbaseband-16-06-2020_15-54-48__empty_TEST_nS-1600_100',...
                               'combinedData_complexbaseband-16-06-2020_15-54-48__empty_TEST_nS-1600_100'};
fileNames_spectrogram       = {'combinedData_spectrogram-16-06-2020_15-54-48__empty_TEST_nS-1600_55',...
                               'combinedData_spectrogram-16-06-2020_15-54-48__empty_TEST_nS-1600_55'};
type                        = 'test';
dim                         = '10x10';
dim_spec                    = '100x10';

%% Create Labels and Images for training
CreateCSV_ComplexBaseband(fileNames_complex, destinationDirectoryName, rootDirectory, type, dim)
CreateCSV_Spectrogram(fileNames_spectrogram, destinationDirectoryName, rootDirectory, type, dim_spec)

