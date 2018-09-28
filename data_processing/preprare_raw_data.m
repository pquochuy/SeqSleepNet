% repare raw data for MASS database

clear all
close all
clc

addpath('./edf_reader/');

raw_data_path = './raw_data/';
if(~exist(raw_data_path, 'dir'))
    mkdir(raw_data_path);
end

% replace location of the MASS database here
data_source = '/media/engs1778/Elements/Dataset/CIBIM/MASS_database/';
subsets = {'SS1','SS2','SS3','SS4','SS5'};
CLASSES = {'W','R','S1','S2','S3'};

for ss = 1 : numel(subsets)
    parentfold = [data_source, subsets{ss}, '/'];
    
    fls = dir([parentfold, '*.edf']);
    fls = arrayfun(@(x) x.name,fls,'UniformOutput',false);

    for f = 1:length(fls)
        
        % Loading data
        [hdr, record] = edfread([parentfold, fls{f}]);
        
        patinfo.ch_orig = cell(2,3);
        patinfo.fs_orig = nan(2,3);
        patinfo.fs = 100;
        patinfo.classes = CLASSES;
        patinfo.chlabels = {'EEG','EOG','EMG'};
        
        %% Getting EEG
        idxeeg = zeros(2,1);
        if any(~cellfun(@isempty, regexp(hdr.label,'(EEGA1)|(EEGA2)')))
            try
                idxeeg(1) = find(~cellfun(@isempty, regexp(hdr.label,'(EEGC4CLE)|(EEGC4LER)')));
                idxeeg(2) = find(~cellfun(@isempty, regexp(hdr.label,'EEGA1CLE')));
            catch
                idxeeg(1) = find(~cellfun(@isempty, regexp(hdr.label,'EEGC3CLE')));
                idxeeg(2) = find(~cellfun(@isempty, regexp(hdr.label,'EEGA2CLE')));
            end
            eeg = diff(record(idxeeg,:))';
        else
            try
                idxeeg = find(~cellfun(@isempty, regexp(hdr.labe,'(EEGC4LER)')));
            catch
                idxeeg = find(~cellfun(@isempty, regexp(hdr.label,'EEGC3LER')));
            end
            eeg = record(idxeeg,:)';
        end
        % sanity check for multiple channels
        if any(idxeeg==0)
            warning('Skipping record, no EEG')
            continue
        end
        % Getting fs
        fseeg = hdr.frequency(idxeeg);
        if length(fseeg)>1
            if fseeg(1) ~= fseeg(2)
                error('Different fs for EEG!?')
            end
        end
        if any(fseeg < 100), error('Low sampling frequency?'), end
        for l = 1:length(idxeeg)
            patinfo.ch_orig(l,1) = cellstr(hdr.label{idxeeg(l)});
            patinfo.fs_orig(l,1) = fseeg(l);
        end
        clear fseeg idxeeg l
        
        
        %% Getting EOG
        idxeog = zeros(2,1);
        try
            idxeog(1) = find(~cellfun(@isempty, regexp(hdr.label,'EOGRightHoriz')));
            idxeog(2) = find(~cellfun(@isempty, regexp(hdr.label,'EOGLeftHoriz')));
        catch
            error('Alternative name')
        end
        
        % sanity check for multiple channels
        if any(idxeog==0)
            warning('Skipping record, no EOG')
            continue
        end
        eog = diff(record(idxeog,:))';
        
        % Getting fspatinfo.fs
        fseog = hdr.frequency(idxeog);
        if length(fseog)>1
            if fseog(1) ~= fseog(2)
                error('Different fs for EOG!?')
            end
        end
        if any(fseog < 100), error('Low sampling frequency?'), end
        for l = 1:length(idxeog)
            patinfo.ch_orig(l,2) = cellstr(hdr.label{idxeog(l)});
            patinfo.fs_orig(l,2) = fseog(l);
        end
        clear fseog idxeog l
        
        
        %% Getting EMG
        if (sum(~cellfun(@isempty, regexp(hdr.label,'EMGChin'))) < 2)    
            idxemg = find(~cellfun(@isempty, regexp(hdr.label,'EMGChin')));
            emg = record(idxemg,:)';
        else
            try
                idxemg(1) = find(~cellfun(@isempty, regexp(hdr.label,'EMGChin1')));
                idxemg(2) = find(~cellfun(@isempty, regexp(hdr.label,'EMGChin2')));
            catch
                error('Alternative name')
            end
            emg = diff(record(idxemg,:))';
        end
        
        % sanity check for multiple channels
        if any(idxemg==0)
            warning('Skipping record, no EMG')
            continue
        end
        
        % Getting fs
        fsemg = hdr.frequency(idxemg);
        if length(fsemg)>1
            if fsemg(1) ~= fsemg(2)
                error('Different fs for EMG!?')
            end
        end
        if any(fsemg < 100), error('Low sampling frequency?'), end
        for l = 1:length(idxemg)
            patinfo.ch_orig(l,3) = cellstr(hdr.label{idxemg(l)});
            patinfo.fs_orig(l,3) = fsemg(l);
        end
        clear fsemg idxemg l
        clear record info idxrow
        
        %% Resampling signals
        fss = round(patinfo.fs_orig(1,:));
        
        %% Preprocessing Filter coefficiens
        Nfir = 100;
        
        % Preprocessing filters
        b_band = fir1(Nfir,[0.3 40].*2/fss(1),'bandpass'); % bandpass
        eeg = filtfilt(b_band,1,eeg);
        
        clear b_notch1 b_notch2 b_band pwrline
        % Preprocessing filters
        b_band = fir1(Nfir,[0.3 40].*2/fss(2),'bandpass'); % bandpass
        eog = filtfilt(b_band,1,eog);
        
        % Preprocessing filters
        pwrline = 50; %Hz
        b_notch1 = fir1(Nfir,[(pwrline-1) (pwrline+1)].*2/fss(3),'stop');
        pwrline = 60; %Hz
        b_notch2 = fir1(Nfir,[(pwrline-1) (pwrline+1)].*2/fss(3),'stop');
        b_band = fir1(Nfir,10.*2/fss(3),'high'); % bandpass
        emg = filtfilt(b_notch1,1,emg);
        emg = filtfilt(b_notch2,1,emg);
        emg = filtfilt(b_band,1,emg);
        
        
        % Resampling to 100 Hz
        eeg = resample(eeg,patinfo.fs,fss(1));
        eog = resample(eog,patinfo.fs,fss(2));
        emg = resample(emg,patinfo.fs,fss(3));
        
        % cut to shortest signal
        rem = find(eeg(end:-1:1) ~= 0,1,'first');
        if (length(eeg) - rem)/patinfo.fs/60/60 < 5 % less than five hours available
            warning('This looks fishy, wrong fs? Skipping..')
            continue
        end
        eeg(end-rem:end) = [];
        rem = find(eog(end:-1:1) ~= 0,1,'first');
        if (length(eog) - rem)/patinfo.fs/60/60 < 5 % less than five hours available
            warning('This looks fishy, wrong fs? Skipping..')
            continue
        end
        eog(end-rem:end) = [];
        rem = find(emg(end:-1:1) ~= 0,1,'first');
        if (length(emg) - rem)/patinfo.fs/60/60 < 5 % less than five hours available
            warning('This looks fishy, wrong fs? Skipping..')
            continue
        end
        emg(end-rem:end) = [];
        % merging elements into matrix
        len = min([length(eeg),length(eog),length(emg)]);
        signals = [eeg(1:len),eog(1:len),emg(1:len)]';
        % Standardizing signals
        for s=1:3
            signals(:,s) = signals(:,s) - nanmean(signals(:,s));
            signals(:,s) = signals(:,s)./nanstd(signals(:,s));
        end
        
        clear eeg eog emg len rem
        %% Figuring out annotations
        annfile = [parentfold, 'annotations/', strip(fls{f}(1:end-7))];
        annfilesaf = [annfile '.saf'];
        annfileedf = dir([annfile '*.edf']);
        annfileedf = ['annotations/' annfileedf.name];
        % Some shell magic
        derror = system(['sed "s/+/\n/g" ' annfilesaf ' > ',parentfold,'annotations/cleanedsaf.txt']);
        derror2 = system(['sed -i "/Sleep stage/!d" ',parentfold,'annotations/cleanedsaf.txt']);
        if (derror || derror2)
            error('Something went wront reading hypnogram')
        end
        
        [fid,msg]=fopen([parentfold, 'annotations/cleanedsaf.txt'],'r','n','UTF-8');
        hypno=textscan(fid,'%[^\n]','delimiter','\n');
        hypno = hypno{1};
        fclose(fid);
        delete([parentfold, 'annotations/cleanedsaf.txt'])
        
        epoch = round(cellfun(@str2double,regexp(hypno,'^\d+\.?\d*','match')));
        nclasses = {'Sleep stage W' 'Sleep stage R' 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 3' 'Sleep stage 4'};
        stages = cellfun(@(x) regexp(x,nclasses),hypno,'UniformOutput',0);
        stages = cell2mat(cellfun(@(x) ~cellfun(@isempty, x), stages,'UniformOutput',0));
        stages = [stages(:,1:4), sum(stages(:,end-1:end),2)]; % converting R&K -> AASM
        stages = stages(:,[1, 3, 4, 5, 2]);  % reordering to match W, N1, N2, N3, R
        rem = ~any(stages,2);  %rows
        epoch(rem,:) = [];
        stages(rem,:) = [];
        clear nclasses hypno fid msg rem
        
        epoch = epoch*patinfo.fs; % including annotation delay on selected epochs
        if epoch(end) > length(signals)
            disp('More annotations than signal. Chop chop!')
            rem = epoch>length(signals);
            epoch(rem) = [];
            stages(rem,:) = [];
        end
        
        labels = logical(stages(1:length(epoch),:));
        clear stages delay ann hdrann hdr i l
        
        data = zeros(length(labels),30*patinfo.fs,3);
        annotinterv = roundn(mean(diff(epoch)/patinfo.fs),1); % annotation interval can be 20s or 30s
        
        if (annotinterv == 20)
            for s = 2:(length(epoch)-1)
                data(s,:,:) = signals(:,epoch(s)-5*patinfo.fs:epoch(s)+(annotinterv+5)*patinfo.fs-1)';
            end
        elseif annotinterv == 30
            for s = 1:(length(epoch))
                data(s,:,:) = signals(:,epoch(s):epoch(s)+annotinterv*patinfo.fs-1)';
            end
        end
        
        
        if any(isnan(data))
            error('NaNs')
        end
        idx = ~any(labels,2)|~any(all(data,2),3);  %rows
        labels(idx,:) = [];
        data(idx,:,:) = [];
        epoch(idx,:) = [];
        if size(data,1) ~= size(labels,1)
            error('Different length for recording..')
        end
        
        %         % Plotting for sanity check
        %         plot(signals')
        %         hold on
        %         for i = 1:length(epoch)
        %             line([epoch(i) epoch(i)],[-100,100],'Color','k')
        %         end
        %         txt = cell(length(epoch),1);
        %         for l = 1:length(epoch)
        %             txt(l) = CLASSES(labels(l,:));
        %         end
        %         text(epoch,100*ones(size(epoch)),txt)
        %         close
        %Saving results
        data = single(data);
        save([raw_data_path, subsets{ss} '_' strip(fls{f}(1:end-7))],'data','labels','patinfo','epoch')
        clear infotxt startsamp lastsamp recstart stage anntm rem data labels patinfo signals r
    end
end
