function GenerateSpectrogram(fileName, directoryName, rootDirectory)
load([rootDirectory '\ProcessedData\' directoryName '\filteredData_' fileName '.mat'])
load([rootDirectory fileName '.mat'])
dim = 10;
nimages = floor(nMeasurements/dim);
if nimages < 1
    error(['Not enough data to form a single spectrogram. Current spectrogram dimensions require at least ' num2str(dim) ' measurements. ' num2str(nMeasurements) ' were received.'])
end

%%
clear img_final1 img_final2 img_final3 img_final4 z
z = 0;
for j = 1:dim
    for k = 1:nimages
        z = z + 1;
        for i = 1:dim
            img_final1{z}(i,:,1) = abs(real(final1{i+dim*k-dim+(j-1)}));
            img_final1{z}(i,:,2) = abs(imag(final1{i+dim*k-dim+(j-1)}));
            img_final2{z}(i,:,1) = abs(real(final2{i+dim*k-dim+(j-1)}));
            img_final2{z}(i,:,2) = abs(imag(final2{i+dim*k-dim+(j-1)}));
            img_final3{z}(i,:,1) = abs(real(final3{i+dim*k-dim+(j-1)}));
            img_final3{z}(i,:,2) = abs(imag(final3{i+dim*k-dim+(j-1)}));
            img_final4{z}(i,:,1) = abs(real(final4{i+dim*k-dim+(j-1)}));
            img_final4{z}(i,:,2) = abs(imag(final4{i+dim*k-dim+(j-1)}));
        end
        img_final234{z}(:,:,1:2) = img_final2{z}(:,:,1:2);
        img_final234{z}(:,:,3:4) = img_final3{z}(:,:,1:2);
        img_final234{z}(:,:,5:6) = img_final4{z}(:,:,1:2);
    end
    nimages = nimages - 1;
end
close all
save([rootDirectory '\ProcessedData\' directoryName '\spectrograms_' fileName '.mat'],'img_final1','img_final2','img_final3','img_final4','img_final234')
end