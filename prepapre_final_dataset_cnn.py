import scipy.io
import pandas as pd
import numpy as np
import os

data_dir = '/gpfs/hpchome/gagand87/project/Data/processed_data/'
os.chdir(data_dir)
eegfiles = list(filter(None, os.popen('find sub*/*/processed_eeg -name sub*.mat').read().split('\n')))


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


# List of classes with more than 200 examples:Counted them from area distribution
# Brodmann area 46 1508
# Brodmann area 4 219
# Optic Tract 345
# Anterior Commissure 246
# Brodmann area 45 224
# Brodmann area 11 1893
# Brodmann area 1 4223
# Brodmann area 3 225
# Brodmann area 42 650
# Brodmann area 19 204
# Brodmann area 21 234
# Lateral Dorsal Nucleus 247
# Brodmann area 10 225
# Midline Nucleus 423
# Brodmann area 38 287
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

# We roll first axis to last as images have channels last and its easier not to modify things with keras
out_x = np.zeros((11153,74,20,43))
o_dfy = pd.DataFrame(np.zeros((11153, 15)))
o_dfy.columns = [area for area in classes_to_keep]
index = 0
for eegfile in eegfiles:

    #eegfilename is like sub001_task001_run001_stimulus_no_18.mat
    #find fmri filename
    fparts = eegfile.split('/')
    fparts[2] = 'avg_per_task_subject'
    fparts[3] = fparts[3].split('.')[0] + '.npy'
    fmri_file_name = '/'.join(fparts)
    eeg_mat = scipy.io.loadmat(eegfile)
    area = ba_areas_list[list(np.load(fmri_file_name)).index(1)]
    if area in classes_to_keep:
        # out_x[index,:,:,:] = np.moveaxis(eeg_mat['normalized_channels'],0,-1)
        out_x[index, :, :, :] = eeg_mat['normalized_channels']
        y = np.zeros(15)
        y[classes_to_keep.index(area)]  = 1
        o_dfy.iloc[index, :] = y
        index  = index + 1

np.save('/gpfs/hpchome/gagand87/project/Data/processed_data/final_dataset/cnn/dataset_sub1to17_EEG_fMRI_cnn_X.pkl',out_x)
o_dfy.to_pickle('/gpfs/hpchome/gagand87/project/Data/processed_data/final_dataset/cnn/dataset_sub1to17_EEG_fMRI_cnn_Y.pkl')