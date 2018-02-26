import pandas as pd
import scipy.io
from nilearn import image
import numpy as np
import os
import multiprocessing as mp
import math
from scipy.fftpack import fft

# Data Dir
data_dir = '/gpfs/hpchome/gagand87/project/Data/'
os.chdir(data_dir)

behav_files = list(filter(None,os.popen('find sub0* -name behavdata.txt').read().split('\n')))
output_df = pd.DataFrame(np.zeros((125,2)))
output_df = output_df.astype('object')
output_df.columns = ['EEG','fMRI']

'''We do follwoing steps on the data:
1. we process the data to correspoding EEG and fMRI
2. Find fft of each EEG signal and save that as 2D numpy array of size 43(electrodes) x 150(freqs of interest) '''
def process(file):
    stimulus = pd.read_csv(file, delim_whitespace=True)
    srt = [.200 if math.isnan(x) else x for x in stimulus['RT']]
    # We add the stimulus run time(SRT) to TrialOnset since the HRF will start after SRT.
    # Also add the +5 for HRF peak
    stimulus_onsets_int = [int(round(x)) + 5 for x in (stimulus['TrialOnset'] + srt)]
    stimulus_onsets = (stimulus['TrialOnset'] * 1000).astype(int)

    fsplits = file.split('/')
    # EEG file
    eeg_file = data_dir + fsplits[0] + '/EEG/' + fsplits[2] + '/EEG_noGA.mat'
    # fMRI file
    fmri_file = data_dir + fsplits[0] + '/BOLD/' + fsplits[2] + '/bold_mcf_brain.nii.gz'

    output_file = data_dir + 'processed_data/' + fsplits[0] + '_' + fsplits[2] + '_df.pkl'
    eeg = scipy.io.loadmat(eeg_file)

    # we take average of 2 nifti images,if stimulus_onsets_int(after adding HRF 5 sec) is odd we take image
    # before 1s than current time and next image. if its even we take current image and one before then we divide
    # by 2 to find the image number,first image is completely acquired at 2 sec(all 32 slices)
    # Carefull to find the indexes as we need to subtract 1 to find 5th image, index should be 4 as indexing starts from 0.
    fMRI_image_indexes = [(int((x - 2) / 2) - 1, int(x / 2) - 1) if x % 2 == 0 else (int((x - 1) / 2) - 1, int((x + 1) / 2) - 1) for
        x in stimulus_onsets_int]

    for index, fMRI_image_index in enumerate(fMRI_image_indexes):
        # print(index, fMRI_image_index)
        output_df.iloc[index, 0] = abs(
            fft(eeg['data_noGA'][:43, stimulus_onsets[index]:stimulus_onsets[index] + 1000])[:, 1:151])
        output_df.iloc[index, 1] = image.mean_img(
            [image.index_img(fmri_file, fMRI_image_index[0]), image.index_img(fmri_file, fMRI_image_index[1])])
    # Save the df
    output_df.to_pickle(output_file)


def main():
    pool = mp.Pool(processes=(10))
    pool.map(process, behav_files)


if __name__ == '__main__':
    main()