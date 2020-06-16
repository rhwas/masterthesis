function Process(fileName,rootDirectory, directoryName)
load([rootDirectory fileName '.mat'])
disp(fileName)
%% 
% Band Pass Filter Design
Fn = Fs/2;
mv = [ 0  0  1  1  0  0];                 % Magnitude Vector
fv = [ 0  35000 37000 43000 45000 Fn]/Fn;          % Frequency Vector
bp  = fir2(2^8, fv, mv);          % Filter Coefficients
delay_bp = mean(grpdelay(bp));
% freqz(bp,1,[],Fs)
r = round(delay_bp+1);
t = TimesBuffer;
TimesBuffer = TimesBuffer(1:end-r+1);

% Determine dimension based on samples per measurement
if length(t) >= 6000
    dim = 20;
elseif length(t) >= 3300
    dim = 15;
elseif length(t) >= 1600
    dim = 10;
else
    error('Sample size too small. <1600')
end

%% Begin processing
for k = 1:nMeasurements
    %% Bandpass Filter
    outbp1 = filter(bp,1,rawTransmitter{k});
    outbp2 = filter(bp,1,rawReciever1{k});
    outbp3 = filter(bp,1,rawReciever2{k});
    outbp4 = filter(bp,1,rawReciever3{k});
    
    %% Convert to Complex-valued signal
    y1 = hilbert(outbp1);
    y2 = hilbert(outbp2);
    y3 = hilbert(outbp3);
    y4 = hilbert(outbp4);
    % Shift to baseband
    for i = 1:nSamples
        y_shift1(i) = y1(i).*exp(-j*2*pi*34500/Fs*i);
        y_shift2(i) = y2(i).*exp(-j*2*pi*34500/Fs*i);
        y_shift3(i) = y3(i).*exp(-j*2*pi*34500/Fs*i);
        y_shift4(i) = y4(i).*exp(-j*2*pi*34500/Fs*i);
    end
    freq1 = fft(y_shift1);
    freq2 = fft(y_shift2);
    freq3 = fft(y_shift3);
    freq4 = fft(y_shift4);


    final_real1{k} = y_shift1;
    final_real2{k} = y_shift2;
    final_real3{k} = y_shift3;
    final_real4{k} = y_shift4;
    final1{k} = freq1(1:dim^2);
    final2{k} = freq2(1:dim^2);
    final3{k} = freq3(1:dim^2);
    final4{k} = freq4(1:dim^2);
    
    
end
if ~exist([rootDirectory '\ProcessedData\' directoryName], 'dir')
    mkdir([rootDirectory '\ProcessedData\' directoryName])
end

save([rootDirectory '\ProcessedData\' directoryName '\filteredData_' fileName '.mat'],'final1','final2','final3','final4','final_real1','final_real2','final_real3','final_real4','y_shift1','y_shift2','y_shift3','y_shift4')
