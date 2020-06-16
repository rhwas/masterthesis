function GenerateComplexBaseband(fileName, directoryName, rootDirectory)
Plotting = false;
load([rootDirectory '\ProcessedData\' directoryName '\filteredData_' fileName '.mat'])
load([rootDirectory fileName '.mat'])
%% Create Image
for k = 1:nMeasurements
    % Create real and imag channels
    channelA1 = abs(real(final1{k}));
    channelA2 = abs(imag(final1{k}));
    channelB1 = abs(real(final2{k}));
    channelB2 = abs(imag(final2{k}));
    channelC1 = abs(real(final3{k}));
    channelC2 = abs(imag(final3{k}));
    channelD1 = abs(real(final4{k}));
    channelD2 = abs(imag(final4{k}));
    channelA1 = (rescale(channelA1,0,1));
    channelA2 = (rescale(channelA2,0,1));
    channelB1 = (rescale(channelB1,0,1));
    channelB2 = (rescale(channelB2,0,1));
    channelC1 = (rescale(channelC1,0,1));
    channelC2 = (rescale(channelC2,0,1));
    channelD1 = (rescale(channelD1,0,1));
    channelD2 = (rescale(channelD2,0,1));
    
    dim = sqrt(length(channelD2));
    
    % Create blue channels for image
    img1{k} = zeros(dim,dim,3);
    img2{k} = zeros(dim,dim,3);
    img3{k} = zeros(dim,dim,3);
    img4{k} = zeros(dim,dim,3);
    
    % Reshape
    channelA1 = reshape(channelA1,[dim dim])';
    channelA2 = reshape(channelA2,[dim dim])';
    channelB1 = reshape(channelB1,[dim dim])';
    channelB2 = reshape(channelB2,[dim dim])';
    channelC1 = reshape(channelC1,[dim dim])';
    channelC2 = reshape(channelC2,[dim dim])';
    channelD1 = reshape(channelD1,[dim dim])';
    channelD2 = reshape(channelD2,[dim dim])';
    
    img1{k}(:,:,1) = channelA1;
    img1{k}(:,:,2) = channelA2;
    img1{k}(:,:,3) = zeros(dim,dim);
    img2{k}(:,:,1) = channelB1;
    img2{k}(:,:,2) = channelB2;
    img2{k}(:,:,3) = zeros(dim,dim);
    img3{k}(:,:,1) = channelC1;
    img3{k}(:,:,2) = channelC2;
    img3{k}(:,:,3) = zeros(dim,dim);
    img4{k}(:,:,1) = channelD1;
    img4{k}(:,:,2) = channelD2;
    img4{k}(:,:,3) = zeros(dim,dim);
    
    % Create images
    final_imgA{k} = xyz2uint16(img1{k});
    final_imgB{k} = xyz2uint16(img2{k});
    final_imgC{k} = xyz2uint16(img3{k});
    final_imgD{k} = xyz2uint16(img4{k});
    
%     final_imgA{k} = img1{k}(:,:,1:2);
%     final_imgB{k} = img2{k}(:,:,1:2);
%     final_imgC{k} = img3{k}(:,:,1:2);
%     final_imgD{k} = img4{k}(:,:,1:2);

    final_imgA{k} = img1{k}(:,:,1:2);
    final_imgBCD{k}(:,:,1:2) = img2{k}(:,:,1:2);
    final_imgBCD{k}(:,:,3:4) = img3{k}(:,:,1:2);
    final_imgBCD{k}(:,:,5:6) = img4{k}(:,:,1:2);
end
if Plotting
    cFull = (-1*Fs)/2:Fs/nSamples:Fs/2-Fs/nSamples;
    cPos = 0:Fs/nSamples:Fs-Fs/nSamples;
    colPlot = ceil(sqrt(nMeasurements));
    rowPlot = ceil(nMeasurements/colPlot);
    % Analog Signal post Bandpass
    figure
    subplot(4,1,1)
    plot(TimesBuffer,outbp1)
    title('ChannelA')
    subplot(4,1,2)
    plot(TimesBuffer,outbp2)
    title('ChannelB')
    subplot(4,1,3)
    plot(TimesBuffer,outbp3)
    title('ChannelC')
    subplot(4,1,4)
    plot(TimesBuffer,outbp4)
    title('ChannelD')
    
    % Plot
    tt = TimesBuffer(1:end-delay_bp);
    sn = y_shift2;
    
    % Plot
    figTitle = ['Bandpass filter - Block Mode Capture Channel' channel_names(1)];
    figure('Name',figTitle, 'NumberTitle', 'off');
    subplot(2,1,1)
    plot(tt,real(sn),'b'); hold on
    plot(tt,imag(sn),'r')
    title('Baseband Signal')
    xlabel('Time')
    legend('Real','Imag')
    subplot(3,1,2)
    plot(tfinal,real(final_real2{k}),'b'); hold on
    plot(tfinal,imag(final_real2{k}),'r')
    title('Resampled Filtered')
    xlabel('Time')
    legend('Real','Imag')
    
    figure
    subplot(rowPlot,colPlot,k)
    image(final_imgA{k})
    figure
    subplot(rowPlot,colPlot,k)
    image(final_imgB{k})
    figure
    subplot(rowPlot,colPlot,k)
    image(final_imgC{k})
    figure
    subplot(rowPlot,colPlot,k)
    image(final_imgD{k})
    
    n = length(final1{k}); % length(x) gives the array length of signal x
    c = 0:Fs/n:Fs-Fs/n; % It generates the frequency series to plot X in frequency domain
    
    figure
    title('ChannelA')
    subplot(2,1,1)
    plot(c,abs(fftshift(real(final1{k}))))
    title('Real')
    subplot(2,1,2)
    plot(c,abs(fftshift(imag(final1{k}))))
    title('Imag')
    figure
    title('ChannelB')
    subplot(2,1,1)
    plot(c,abs(fftshift(real(final2{k}))))
    title('Real')
    subplot(2,1,2)
    plot(c,abs(fftshift(imag(final2{k}))))
    title('Imag')
    figure
    title('ChannelC')
    subplot(2,1,1)
    plot(c,abs(fftshift(real(final3{k}))))
    title('Real')
    subplot(2,1,2)
    plot(c,abs(fftshift(imag(final3{k}))))
    title('Imag')
    figure
    title('ChannelD')
    subplot(2,1,1)
    plot(c,abs(fftshift(real(final4{k}))))
    title('Real')
    subplot(2,1,2)
    plot(c,abs(fftshift(imag(final4{k}))))
    title('Imag')
end
save([rootDirectory '\ProcessedData\' directoryName '\complexbaseband_' fileName '.mat'],'final_imgA','final_imgB','final_imgC','final_imgD','final_imgBCD')
