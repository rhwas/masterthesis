function CreateCSV_Spectrogram(fileNames, directoryName, rootDirectory, type, dim_spec)
labels  = 0:1:length(fileNames)-1;

folderLocation = ['E:\Chalmers\TestProject\TrainingData\' type '\spectrogram\' dim_spec '_' directoryName];
if ~exist(folderLocation, 'dir')
    mkdir(folderLocation)
end
figure; colormap gray; cmap = colormap; close all
sum = 0; object = []; id = [];
for j = 1:length(labels)
    load([rootDirectory '\ProcessedData\' directoryName '\' fileNames{j} '.mat'])
    disp('Spectrogram')
    nMeasurements{j} = length(imgsBCD)
    if j>1
        sum = sum + nMeasurements{j-1};
    end
    for k = 1:nMeasurements{j}
        if j == 1
%             imgsB{k}(:,:,3) = zeros(size(imgsB{k}(:,:,1)));
%             imgsB{k}(:,:,1) = rescale(imgsB{k}(:,:,1),0,1);
%             imgsB{k}(:,:,2) = rescale(imgsB{k}(:,:,2),0,1);
%             imgsC{k}(:,:,3) = zeros(size(imgsC{k}(:,:,1)));
%             imgsC{k}(:,:,1) = rescale(imgsC{k}(:,:,1),0,1);
%             imgsC{k}(:,:,2) = rescale(imgsC{k}(:,:,2),0,1);
%             imgsD{k}(:,:,3) = zeros(size(imgsD{k}(:,:,1)));
%             imgsD{k}(:,:,1) = rescale(imgsD{k}(:,:,1),0,1);
%             imgsD{k}(:,:,2) = rescale(imgsD{k}(:,:,2),0,1);
%             imwrite(imgsB{k},cmap,[folderLocation '\img' num2str(k-1) '_b.png'],'png')
%             imwrite(imgsC{k},cmap,[folderLocation '\img' num2str(k-1) '_c.png'],'png')
%             imwrite(imgsD{k},cmap,[folderLocation '\img' num2str(k-1) '_d.png'],'png')
%             exportimgs = imgsB{k};
%             save([folderLocation '\img' num2str(k-1) '_b.mat'],'exportimgs')
%             exportimgs = imgsC{k};
%             save([folderLocation '\img' num2str(k-1) '_c.mat'],'exportimgs')
%             exportimgs = imgsD{k};
%             save([folderLocation '\img' num2str(k-1) '_d.mat'],'exportimgs')
            exportimgs = imgsBCD{k};
            save([folderLocation '\img' num2str(k-1) '_bcd.mat'],'exportimgs')
        else
%             imgsB{k}(:,:,3) = zeros(size(imgsB{k}(:,:,1)));
%             imgsB{k}(:,:,1) = rescale(imgsB{k}(:,:,1),0,1);
%             imgsB{k}(:,:,2) = rescale(imgsB{k}(:,:,2),0,1);
%             imgsC{k}(:,:,3) = zeros(size(imgsC{k}(:,:,1)));
%             imgsC{k}(:,:,1) = rescale(imgsC{k}(:,:,1),0,1);
%             imgsC{k}(:,:,2) = rescale(imgsC{k}(:,:,2),0,1);
%             imgsD{k}(:,:,3) = zeros(size(imgsD{k}(:,:,1)));
%             imgsD{k}(:,:,1) = rescale(imgsD{k}(:,:,1),0,1);
%             imgsD{k}(:,:,2) = rescale(imgsD{k}(:,:,2),0,1);
%             imwrite(imgsB{k},cmap,[folderLocation '\img' num2str(k+sum-1) '_b.png'],'png')
%             imwrite(imgsC{k},cmap,[folderLocation '\img' num2str(k+sum-1) '_c.png'],'png')
%             imwrite(imgsD{k},cmap,[folderLocation '\img' num2str(k+sum-1) '_d.png'],'png')
%             exportimgs = imgsB{k};
%             save([folderLocation '\img' num2str(k+sum-1) '_b.mat'],'exportimgs')
%             exportimgs = imgsC{k};
%             save([folderLocation '\img' num2str(k+sum-1) '_c.mat'],'exportimgs')
%             exportimgs = imgsD{k};
%             save([folderLocation '\img' num2str(k+sum-1) '_d.mat'],'exportimgs')
            exportimgs = imgsBCD{k};
            save([folderLocation '\img' num2str(k+sum-1) '_bcd.mat'],'exportimgs')
        end
    end
    object = [object;labels(j)*ones(nMeasurements{j},1)];
end
sumsum = 0;
for j = 1:length(labels)
    sumsum = sumsum + nMeasurements{j};
end
id = 0:1:sumsum-1;
A = [id' object];
T = array2table(A);
T.Properties.VariableNames(1:2) = {'id','object'};
writetable(T,['E:\Chalmers\TestProject\TrainingData\' type '_labels_spec_' directoryName '.txt'])
