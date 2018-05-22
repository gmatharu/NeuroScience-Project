import scipy.io
import pandas as pd
import numpy as np
import os
ba_areas_list = ['Amygdala',
                 'Anterior Commissure',
                 'Anterior Nucleus',
                 'Background',
                 'Brodmann area 1',
                 'Brodmann area 10',
                 'Brodmann area 11',
                 'Brodmann area 13',
                 'Brodmann area 17',
                 'Brodmann area 18',
                 'Brodmann area 19',
                 'Brodmann area 2',
                 'Brodmann area 20',
                 'Brodmann area 21',
                 'Brodmann area 22',
                 'Brodmann area 23',
                 'Brodmann area 24',
                 'Brodmann area 25',
                 'Brodmann area 27',
                 'Brodmann area 28',
                 'Brodmann area 29',
                 'Brodmann area 3',
                 'Brodmann area 30',
                 'Brodmann area 31',
                 'Brodmann area 32',
                 'Brodmann area 33',
                 'Brodmann area 34',
                 'Brodmann area 35',
                 'Brodmann area 36',
                 'Brodmann area 37',
                 'Brodmann area 38',
                 'Brodmann area 39',
                 'Brodmann area 4',
                 'Brodmann area 40',
                 'Brodmann area 41',
                 'Brodmann area 42',
                 'Brodmann area 43',
                 'Brodmann area 44',
                 'Brodmann area 45',
                 'Brodmann area 46',
                 'Brodmann area 47',
                 'Brodmann area 5',
                 'Brodmann area 6',
                 'Brodmann area 7',
                 'Brodmann area 8',
                 'Brodmann area 9',
                 'Caudate Body',
                 'Caudate Head',
                 'Caudate Tail',
                 'Corpus Callosum',
                 'Dentate',
                 'Hippocampus',
                 'Hypothalamus',
                 'Lateral Dorsal Nucleus',
                 'Lateral Geniculum Body',
                 'Lateral Globus Pallidus',
                 'Lateral Posterior Nucleus',
                 'Mammillary Body',
                 'Medial Dorsal Nucleus',
                 'Medial Geniculum Body',
                 'Medial Globus Pallidus',
                 'Midline Nucleus',
                 'Optic Tract',
                 'Pulvinar',
                 'Putamen',
                 'Red Nucleus',
                 'Substania Nigra',
                 'Subthalamic Nucleus',
                 'Ventral Anterior Nucleus',
                 'Ventral Lateral Nucleus',
                 'Ventral Posterior Lateral Nucleus',
                 'Ventral Posterior Medial Nucleus']





classes_to_keep = ['Brodmann area 46',
'Brodmann area 4',
'Optic Tract',
'Anterior Commissure',
'Brodmann area 45',
'Brodmann area 11',
'Brodmann area 1',
'Brodmann area 3',
'Brodmann area 42',
'Brodmann area 19',
'Brodmann area 21',
'Lateral Dorsal Nucleus',
'Brodmann area 10',
'Midline Nucleus',
'Brodmann area 38']
ind = 0
data_dir = '/home/hpc_gagand87/beamformer/data/'
os.chdir(data_dir)

behav_files = list(filter(None, os.popen('find sub*/behav/*/ -name behavdata.txt').read().split('\n')))
o_dfx = pd.DataFrame(np.zeros((12750,43001)))
print(behav_files)
o_dfx.columns = ['intentsity_' + str(i) for i in range(43000)] + ['area']

for file in behav_files:
    stimulus = pd.read_csv(file, delim_whitespace=True)
    stimulus_onsets = (stimulus['TrialOnset'] * 1000).astype(int)

    fsplits = file.split('/')
    # EEG file
    eeg_file = data_dir + fsplits[0] + '/EEG/' + fsplits[2] + '/EEG_noGA.mat'
    eeg = scipy.io.loadmat(eeg_file)

    for index, stimulus_onset in enumerate(stimulus_onsets):
        # prepare the data structure
        signal = eeg['data_noGA'][:43, stimulus_onset:stimulus_onset + 1000]
        fmri_file_name = data_dir +fsplits[0] + '/'+fsplits[2]+'/avg_per_task_subject/'+fsplits[0]+'_'+fsplits[2]+ '_stimulus_no_'+str(index)+'.npy'
        print(eeg_file,file,fmri_file_name)
        cba = ba_areas_list[list(np.load(fmri_file_name)).index(1)]
        if cba in classes_to_keep:
            o_dfx.iloc[ind, :43000] = np.reshape(signal, 43000)
            o_dfx.iloc[ind, 43000:] = cba
            ind = ind + 1
        else:
            o_dfx.iloc[ind, :43000] = np.zeros(43000)
            o_dfx.iloc[ind, 43000:] = 'NA'
            ind = ind + 1
o_dfx.to_pickle('/home/hpc_gagand87/beamformer/data/dataset/dataset_sub1to17.pkl')