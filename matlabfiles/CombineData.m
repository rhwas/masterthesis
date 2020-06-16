close all; clear all; clc;
%% Consoladates the Complex baseband and Spectrogram input and creates the three construction methods
rootDirectory   = 'E:\Chalmers\TestProject\'; % Data location
destinationDirectoryName    = 'w30';
samples         = {'1600','3300','6000'};
for j = 1:3
    tag                         = ['_empty_TEST_nS-' samples{j}];
    fileList                    = dir([rootDirectory '*' tag '*.mat']);
    for y = 1:length(fileList)
        [filepath, name, ext] = fileparts(fileList(y).name);
        fileNames{y} = name;
    end
    % fileNames                   = {'data-09-04-2020_16-06-11_squareRotate10cm20deg_nS-1800_nM-2000'};
    
    %% Complex Baseband
    for i = 1:length(fileNames)
        load([rootDirectory '\ProcessedData\' destinationDirectoryName '\complexbaseband_' fileNames{i} '.mat'])
        if i == 1
            imgsB = final_imgB;
            imgsC = final_imgC;
            imgsD = final_imgD;
            imgsBCD = final_imgBCD;
        else
            imgsB = [imgsB final_imgB];
            imgsC = [imgsC final_imgC];
            imgsD = [imgsD final_imgD];
            imgsBCD = [imgsBCD final_imgBCD];
        end
    end
    N = length(imgsB);
    fileName = ['combinedData_complexbaseband-',datestr(now,'dd-mm-yyyy_HH-MM-SS'),'_' tag];
    save([rootDirectory '\ProcessedData\' destinationDirectoryName '\' fileName '_' num2str(N) '.mat'],'imgsB','imgsC','imgsD','imgsBCD')
    clear imgsB imgsC imgsD imgsBCD final_imgB final_imgC final_imgD final_imgBCD
    
    %% Spectrogram
    for i = 1:length(fileNames)
        load([rootDirectory '\ProcessedData\' destinationDirectoryName '\spectrograms_' fileNames{i} '.mat'])
%         final_imgB = img_final2;
%         final_imgC = img_final3;
%         final_imgD = img_final4;
        final_imgBCD = img_final234;
        if i == 1
%             imgsB = final_imgB;
%             imgsC = final_imgC;
%             imgsD = final_imgD;
            imgsBCD = final_imgBCD;
        else
%             imgsB = [imgsB final_imgB];
%             imgsC = [imgsC final_imgC];
%             imgsD = [imgsD final_imgD];
            imgsBCD = [imgsBCD final_imgBCD];
        end
    end
    N = length(imgsBCD);
    fileName = ['combinedData_spectrogram-',datestr(now,'dd-mm-yyyy_HH-MM-SS'),'_' tag];
%     save([rootDirectory '\ProcessedData\' destinationDirectoryName '\' fileName '_' num2str(N) '.mat'],'imgsB','imgsC','imgsD')
    save([rootDirectory '\ProcessedData\' destinationDirectoryName '\' fileName '_' num2str(N) '.mat'],'imgsBCD')
end