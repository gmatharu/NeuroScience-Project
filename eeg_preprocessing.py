import scipy.io
import pandas as pd
import numpy as np
from mne import create_info, EpochsArray
from mne.time_frequency import tfr_morlet
import os

data_dir = '/gpfs/hpchome/gagand87/project/Data/'
os.chdir(data_dir)
behav_files = list(filter(None, os.popen('find ./sub001/behav/task001_run001/ -name behavdata.txt').read().split('\n')))

for file in behav_files:
    stimulus = pd.read_csv(file, delim_whitespace=True)
    stimulus_onsets = (stimulus['TrialOnset'] * 1000).astype(int)

    fsplits = file.split('/')
    # EEG file
    eeg_file = data_dir + fsplits[0] + '/EEG/' + fsplits[2] + '/EEG_noGA.mat'
    output_dir = data_dir + 'processed_data/' + fsplits[0] + '/' + fsplits[2]+ '/processed_eeg/'
    os.makedirs(output_dir, exist_ok=True)
    eeg = scipy.io.loadmat(eeg_file)

    for index, stimulus_onset in enumerate(stimulus_onsets):
        # prepare the data structure
        signal = eeg['data_noGA'][:43, stimulus_onset - 500:stimulus_onset + 1000]
        info = create_info(ch_names=['CH' + str(i) for i in range(1, 44)], sfreq=1000,
                           ch_types=['eeg' for i in range(1, 44)])
        signal = signal.reshape(1, signal.shape[0], signal.shape[1])
        epochs = EpochsArray(signal, info)
        freqs = np.arange(4., 151., 2.)
        n_cycles = freqs / 3.
        power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
        # outfile = 'sub001_task001_run001_stimulus_no_'+str(index)+'.mat'
        outfname = fsplits[0] + '_' + fsplits[2] + '_stimulus_no_' + str(index)+'.mat'
        outfile = output_dir + outfname
        output= np.zeros((43, 74, 1000))
        for channel in np.arange(0,43,1):
            baseline_means = np.mean(power.data[channel, :, 0:400], 1)
            baseline_means = np.swapaxes(np.tile(baseline_means, (1000, 1)), 0, 1)
            normalized_signal = power.data[channel, :, 500:] / baseline_means
            output[channel,:,:]=np.flipud(normalized_signal)
        outnorm = np.zeros((43, 74, 20))
        for freq in range(74):
            for index, time in enumerate(range(0, 951, 50)):
                outnorm[:, freq, index] = output[:, freq, time:time + 50].mean(axis=1)
        scipy.io.savemat(outfile,{'normalized_channels':outnorm})