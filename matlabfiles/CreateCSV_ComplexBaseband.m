function CreateCSV_ComplexBaseband(fileNames, directoryName, rootDirectory, type, dim)

labels  = 0:1:length(fileNames)-1;

folderLocation = ['E:\Chalmers\TestProject\TrainingData\' type '\complexbaseband\' dim '_' directoryName];
if ~exist(folderLocation, 'dir')
    mkdir(folderLocation)
end

sum = 0; object = []; id = [];
for j = 1:length(labels)
    load([rootDirectory '\ProcessedData\' directoryName '\' fileNames{j} '.mat'])
    disp('Complexbaseband')
    nMeasurements{j} = length(imgsB)
    if j>1
        sum = sum + nMeasurements{j-1};
    end
    for k = 1:nMeasurements{j}
        if j == 1
            imwrite(imgsB{k},[folderLocation '\img' num2str(k-1) '_b.png'],'png')
            imwrite(imgsC{k},[folderLocation '\img' num2str(k-1) '_c.png'],'png')
            imwrite(imgsD{k},[folderLocation '\img' num2str(k-1) '_d.png'],'png')
            exportimgs = imgsB{k};
            save([folderLocation '\img' num2str(k-1) '_b.mat'],'exportimgs')
            exportimgs = imgsC{k};
            save([folderLocation '\img' num2str(k-1) '_c.mat'],'exportimgs')
            exportimgs = imgsD{k};
            save([folderLocation '\img' num2str(k-1) '_d.mat'],'exportimgs')
            exportimgs = imgsBCD{k};
            save([folderLocation '\img' num2str(k-1) '_bcd.mat'],'exportimgs')
        else
            imwrite(imgsB{k},[folderLocation '\img' num2str(k+sum-1) '_b.png'],'png')
            imwrite(imgsC{k},[folderLocation '\img' num2str(k+sum-1) '_c.png'],'png')
            imwrite(imgsD{k},[folderLocation '\img' num2str(k+sum-1) '_d.png'],'png')
            exportimgs = imgsB{k};
            save([folderLocation '\img' num2str(k+sum-1) '_b.mat'],'exportimgs')
            exportimgs = imgsC{k};
            save([folderLocation '\img' num2str(k+sum-1) '_c.mat'],'exportimgs')
            exportimgs = imgsD{k};
            save([folderLocation '\img' num2str(k+sum-1) '_d.mat'],'exportimgs')
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
writetable(T,['E:\Chalmers\TestProject\TrainingData\' type '_labels_' directoryName '.txt'])
