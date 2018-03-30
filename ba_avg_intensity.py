import pandas as pd
import math
import os
import bz2
import pickle
import numpy as np

data_dir = '/gpfs/hpchome/gagand87/project/Data/'
os.chdir(data_dir)

ba_areas = ['Amygdala',
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
out_dict = {}
behav_files = list(filter(None, os.popen('find sub*/behav/task*/  -name behavdata.txt').read().split('\n')))
for file in behav_files:
    stimulus = pd.read_csv(file, delim_whitespace=True)
    srt = [.200 if math.isnan(x) else x for x in stimulus['RT']]
    # We add the stimulus run time(SRT) to TrialOnset since the HRF will start after SRT.
    # Also add the +5 for HRF peak
    stimulus_onsets_int = [int(round(x)) + 5 for x in (stimulus['TrialOnset'] + srt)]
    stimulus_onsets = (stimulus['TrialOnset'] * 1000).astype(int)

    fsplits = file.split('/')

    # we take average of 2 nifti images,if stimulus_onsets_int(after adding HRF 5 sec) is odd we take image
    # before 1s than current time and next image. if its even we take current image and one before then we divide
    # by 2 to find the image number,first image is completely acquired at 2 sec(all 32 slices)
    # Carefull to find the indexes as we need to subtract 1 to find 5th image, index should be 4 as indexing starts from 0.
    fMRI_image_indexes = [
        (int((x - 2) / 2) - 1, int(x / 2) - 1) if x % 2 == 0 else (int((x - 1) / 2) - 1, int((x + 1) / 2) - 1) for
        x in stimulus_onsets_int]
    print(fMRI_image_indexes)
    for stimno,indexes in enumerate(fMRI_image_indexes):
        ba_mean_inten_out_file = 'processed_data/'+fsplits[0]+'/'+ fsplits[2]+'/ba_avg_intensity/'+ fsplits[0] + '_' + fsplits[2] + '_stimulus_no_' + str(stimno)+'.pkl.bz2'
        out_dir = os.path.dirname(ba_mean_inten_out_file)
        os.makedirs(out_dir, exist_ok=True)
        filename1 = 'processed_data/' + fsplits[0] + '/' + fsplits[
            2] + '/BA_Intensity/' + 'bold_mcf_brain_BA_Intensity' + str(indexes[0]) + '.pkl'
        filename2 = 'processed_data/' + fsplits[0] + '/' + fsplits[
            2] + '/BA_Intensity/' + 'bold_mcf_brain_BA_Intensity' + \
                    str(indexes[1]) + '.pkl'
        with bz2.BZ2File(filename1, 'r') as handle:
            file1 = pickle.load(handle)
        with bz2.BZ2File(filename2, 'r') as handle:
            file2 = pickle.load(handle)
        for ba_area in ba_areas:
            f1ba_val = file1[ba_area]
            f2ba_val = file2[ba_area]
            if len(f1ba_val) == 0:
                f1ba_val = 0
            if len(f2ba_val) == 0:
                f2ba_val = 0
            f1ba_mean = np.mean(f1ba_val)
            f2ba_mean = np.mean(f2ba_val)
            out_mean = np.mean([f1ba_mean,f2ba_mean])
            out_dict[ba_area] = out_mean
        with bz2.BZ2File(ba_mean_inten_out_file, 'w') as handle:
            pickle.dump(out_dict, handle)