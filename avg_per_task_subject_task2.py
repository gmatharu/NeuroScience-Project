import os
import pandas as pd
import numpy as np
import bz2, pickle

data_dir = '/gpfs/hpchome/gagand87/project/Data/processed_data'
os.chdir(data_dir)
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
for i in range(1, 18):
    ba_areas = {'Amygdala': [],
                'Anterior Commissure': [],
                'Anterior Nucleus': [],
                'Background': [],
                'Brodmann area 1': [],
                'Brodmann area 10': [],
                'Brodmann area 11': [],
                'Brodmann area 13': [],
                'Brodmann area 17': [],
                'Brodmann area 18': [],
                'Brodmann area 19': [],
                'Brodmann area 2': [],
                'Brodmann area 20': [],
                'Brodmann area 21': [],
                'Brodmann area 22': [],
                'Brodmann area 23': [],
                'Brodmann area 24': [],
                'Brodmann area 25': [],
                'Brodmann area 27': [],
                'Brodmann area 28': [],
                'Brodmann area 29': [],
                'Brodmann area 3': [],
                'Brodmann area 30': [],
                'Brodmann area 31': [],
                'Brodmann area 32': [],
                'Brodmann area 33': [],
                'Brodmann area 34': [],
                'Brodmann area 35': [],
                'Brodmann area 36': [],
                'Brodmann area 37': [],
                'Brodmann area 38': [],
                'Brodmann area 39': [],
                'Brodmann area 4': [],
                'Brodmann area 40': [],
                'Brodmann area 41': [],
                'Brodmann area 42': [],
                'Brodmann area 43': [],
                'Brodmann area 44': [],
                'Brodmann area 45': [],
                'Brodmann area 46': [],
                'Brodmann area 47': [],
                'Brodmann area 5': [],
                'Brodmann area 6': [],
                'Brodmann area 7': [],
                'Brodmann area 8': [],
                'Brodmann area 9': [],
                'Caudate Body': [],
                'Caudate Head': [],
                'Caudate Tail': [],
                'Corpus Callosum': [],
                'Dentate': [],
                'Hippocampus': [],
                'Hypothalamus': [],
                'Lateral Dorsal Nucleus': [],
                'Lateral Geniculum Body': [],
                'Lateral Globus Pallidus': [],
                'Lateral Posterior Nucleus': [],
                'Mammillary Body': [],
                'Medial Dorsal Nucleus': [],
                'Medial Geniculum Body': [],
                'Medial Globus Pallidus': [],
                'Midline Nucleus': [],
                'Optic Tract': [],
                'Pulvinar': [],
                'Putamen': [],
                'Red Nucleus': [],
                'Substania Nigra': [],
                'Subthalamic Nucleus': [],
                'Ventral Anterior Nucleus': [],
                'Ventral Lateral Nucleus': [],
                'Ventral Posterior Lateral Nucleus': [],
                'Ventral Posterior Medial Nucleus': []}
    if i < 10:
        subject = 'sub00' + str(i)
    else:
        subject = 'sub0' + str(i)
    fmrifiles = list(
        filter(None, os.popen('find ' + subject + '/task002*/ba_avg_intensity/ -name "*.pkl.bz2"').read().split('\n')))
    for index, fmrifile in enumerate(fmrifiles):
        with bz2.BZ2File(fmrifile, 'r') as handle:
            baai = pickle.load(handle)
        for area in ba_areas:
            ba_areas[area].append(baai[area])
    outfile = '/gpfs/hpchome/gagand87/project/Data/processed_data/avg_per_sub/' + subject + '_avg_areas_task2.pkl.bz2'
    mean_dict = {}
    for baarea in ba_areas.keys():
        mean_dict[baarea] = np.mean(ba_areas[baarea])
    out_dict = {}
    output_df = pd.DataFrame(np.zeros((375, 1)))
    output_df = output_df.astype('object')
    output_df.columns = ['avg_int']
    for index, fmrifile in enumerate(fmrifiles):
        fname_parts = fmrifile.split('/')
        fn = fname_parts[3].split('.')[0]
        outfname = data_dir +'/'+ fname_parts[0] + '/' + fname_parts[1] + '/avg_per_task_subject/' + fn
        out_dir = os.path.dirname(outfname)
        os.makedirs(out_dir, exist_ok=True)
        with bz2.BZ2File(fmrifile, 'r') as handle:
            baai = pickle.load(handle)
        for area in baai:
            out_dict[area] = baai[area] / mean_dict[area]
        y = np.zeros(72)
        max_value = max(out_dict.values())
        max_key = [key for key, value in out_dict.items() if value == max_value]
        y[ba_areas_list.index(max_key[0])] = 1
        np.save(outfname,y)
        final_dict = {}
        for ind, val in enumerate(out_dict.keys()):
            final_dict[val] = y[ind]
        output_df.iloc[index, 0] = [final_dict]
    output_df.to_pickle(outfile)
    print(subject)
