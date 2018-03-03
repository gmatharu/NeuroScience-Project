import os
import pandas as pd
import numpy as np
import scipy.io
import bz2, pickle,h5py

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

data_dir = '/gpfs/hpchome/gagand87/project/Data/processed_data/'
os.chdir(data_dir)
output_df = pd.DataFrame(np.zeros((12750, 2)))
output_df = output_df.astype('object')
output_df.columns = ['EEG', 'fMRI']
eegfiles = list(filter(None, os.popen('find sub0*/*/processed_eeg -name *.mat').read().split('\n')))
for index, eegfile in enumerate(eegfiles):
    fparts = eegfile.split('/')
    fparts[2] = 'ba_avg_intensity'
    fparts[3] = fparts[3].split('.')[0] + '.pkl.bz2'
    label_file_name = '/'.join(fparts)
    print(label_file_name)
    eeg_mat = scipy.io.loadmat(eegfile)
    output_df.iloc[index, 0] = eeg_mat['normalized_channels']
    y = np.zeros(72)
    with bz2.BZ2File(label_file_name, 'r') as handle:
        baai = pickle.load(handle)
    max_value = max(baai.values())
    print(max_value)
    max_key = [key for key, value in baai.items() if value == max_value]
    y[ba_areas_list.index(max_key[0])] = 1
    print(y,max_key)
    output_df.iloc[index, 1] = y
output_df.to_pickle('/gpfs/hpchome/gagand87/project/Data/final_dataset/dataset.pkl')
