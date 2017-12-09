import pandas as pd
import scipy.io
from nilearn import image
import numpy as np
import os
import multiprocessing as mp

#Data Dir
data_dir='/gpfs/hpchome/gagand87/project/Data/'
os.chdir(data_dir)

behav_files=list(filter(None,os.popen('find sub0* -name behavdata.txt').read().split('\n')))
output_df = pd.DataFrame(np.zeros((125,2)))
output_df = output_df.astype('object')
output_df.columns=['EEG','fMRI']


def process(file):
    stimulus = pd.read_csv(file,delim_whitespace=True)
    stimulus_onsets_int = (stimulus['TrialOnset']).astype(int)
    stimulus_onsets = (stimulus['TrialOnset'] * 1000).astype(int)

    fsplits = file.split('/')
    #EEG file
    eeg_file=data_dir+fsplits[0]+'/EEG/'+fsplits[2]+'/EEG_noGA.mat'
    #fMRI file
    fmri_file=data_dir + fsplits[0] + '/BOLD/' + fsplits[2] + '/bold_mcf_brain.nii.gz'

    output_file = data_dir+fsplits[0]+fsplits[2]+'_df.pkl'
    eeg = scipy.io.loadmat(eeg_file)

    # we move 2 seconds in time if the round down stimulus onset is even, if its odd, we move 3 sec ahead
    # then we divide by 2 to find the image number, currently we assume first image at 2 sec
    fMRI_image_indexes = [int((x + 2) / 2) if x % 2 == 0 else int((x + 3) / 2) for x in stimulus_onsets_int]

    for index, fMRI_image_index in enumerate(fMRI_image_indexes):
        output_df.iloc[index, 0] = eeg['data_noGA'][:43, stimulus_onsets[index]:stimulus_onsets[index] + 1000]
        output_df.iloc[index, 1] = image.index_img(fmri_file,fMRI_image_index)
    output_df.to_pickle(output_file)


def main():
    pool = mp.Pool(processes=(10))
    pool.map(process, behav_files)


if __name__ == '__main__':
    main()