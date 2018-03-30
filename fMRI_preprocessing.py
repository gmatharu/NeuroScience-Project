"""The purpose of this file is to take each subjects anatomical image and BOLD 4D volume, and do following steps:
1.coregister and spatially normalize using fsl Flirt
2.voxel to MNI to Tal affine transformation
3.Tal coordinates to BA

Output:
We save each functional volume as a pickled dictionary, each key=Brodmann area, value= intensities for these BA's:
example:
'Brodmann Area' 1:[1,2,3,4]

we must have fsl installed on the server for coregisteration add normalization
"""

from nilearn import image
import os
import pickle
import numpy as np
from nilearn.image import coord_transform
# for running shell script to coreg and norm
import subprocess
import shlex
import bz2

data_dir = '/gpfs/hpchome/gagand87/project/Data/'
os.chdir(data_dir)

result_dir = '/gpfs/hpchome/gagand87/project/Data/processed_data/'

def create_ind_vols(fourdnifti=[]):
    for indvol in fourdnifti:
        vol_name_parts = indvol.split('/')
        vol = image.load_img(indvol)
        for imind in range(vol.shape[3]):
            out_file_name = result_dir + vol_name_parts[0] + '/' + vol_name_parts[
                2] + "/IndFMRIs/bold_mcf_brain_" + str(imind) + ".nii.gz"
            im = image.index_img(vol,imind)
            out_dir = os.path.dirname(out_file_name)
            os.makedirs(out_dir, exist_ok=True)
            im.to_filename(out_file_name)
    return "success"


def coreg():
    #chage subject and task run below for ind subjects
    indvols=list(filter(None, os.popen('find processed_data/sub002/task001_run001/IndFMRIs/ -name *.nii.gz').read().split('\n')))
    for ind,fmri_vol in enumerate(indvols):
        # vol = image.load_img(fmri_vol)
        # name comes in as 'processed_data/sub001/task001_run002/IndFMRIs/bold_mcf_brain_71.nii.gz'
        fmri_vol_name_parts = fmri_vol.split('/')
        # high res image name
        anat = data_dir + fmri_vol_name_parts[1] + '/anatomy/highres001_brain.nii.gz'
        cornorm_out_fname = fmri_vol_name_parts[4].split('_')[3]
        cornorm_out = result_dir + fmri_vol_name_parts[1] + '/' + fmri_vol_name_parts[
            2] + "/Norm/bold_mcf_brainCorNorm_" + cornorm_out_fname

        # make sure to create target path
        out_dir = os.path.dirname(cornorm_out)
        out_fname = os.path.basename(cornorm_out)
        os.makedirs(out_dir, exist_ok=True)

        command_name = '/gpfs/hpchome/gagand87/project/coreg_norm_Flirt.sh ' + out_dir + ' ' + anat + ' ' + \
                       '/gpfs/hpchome/gagand87/project/fsl/data/standard/MNI152_T1_2mm_brain' + ' ' + fmri_vol+ ' ' + out_fname
        subprocess.call(shlex.split(command_name))
    return "success"


def vol_ba():
    cor_vols = list(
        filter(None, os.popen('find processed_data -name bold_mcf_brainCorNorm_*.nii.gz').read().split('\n')))
    # MNI to TAL Transform
    affine = np.zeros((4, 4))
    affine[0] = [0.88, 0, 0, -0.8]
    affine[1] = [0, 0.97, 0, -3.32]
    affine[2] = [0, 0.05, 0.88, -0.44]
    affine[3] = [0, 0, 0, 1]

    with open('/gpfs/hpchome/gagand87/project/tal_to_BA.pkl', 'rb') as handle:
        tal_to_ba = pickle.load(handle)

    for ind, vol in enumerate(cor_vols):
        ba_intensity = {'Amygdala': [],
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
        vol_name_parts = vol.split('/')
        BA_intensity_file_name = vol_name_parts[0] + '/' + vol_name_parts[1] + '/' + vol_name_parts[
            2] + '/' + 'BA_Intensity/bold_mcf_brain_BA_Intensity' + str(ind) + '.pkl'
        fmri = image.load_img(vol)
        fmri_data = fmri.get_data()
        for i in range(fmri.shape[0]):
            for j in range(fmri.shape[1]):
                for k in range(fmri.shape[2]):
                    mni_coord = coord_transform(i, j, k, fmri.affine)
                    tal_coord = coord_transform(mni_coord[0], mni_coord[1], mni_coord[2], affine)
                    # Since tal coordinates range is (-70.0, -102.0, -42.0) to (70.0, 69.0, 67.0), we ignore the rest of the coordinates
                    if -70 <= tal_coord[0] <= 70 and -102 <= tal_coord[1] <= 69 and -42 <= tal_coord[2] <= 67:
                        t_coord = [0] * 3
                        t_coord[0] = round(tal_coord[0])
                        t_coord[1] = round(tal_coord[1])
                        t_coord[2] = round(tal_coord[2])
                        ba_intensity[tal_to_ba[(t_coord[0], t_coord[1], t_coord[2])]].append(fmri_data[i, j, k])

        with bz2.BZ2File(BA_intensity_file_name, 'wb') as handle:
            pickle.dump(ba_intensity, handle)


if __name__ == '__main__':
    fmri_vols = list(filter(None, os.popen('find sub0* -name bold_mcf_brain.nii.gz').read().split('\n')))
    code = create_ind_vols(fmri_vols)
    code= 'success'
    if code == "success":
        print("Indv volumes creation success")
        coreg_out = coreg()
    else:
        print("Issues with Ind vol creation")
    if coreg_out == "success":
        print("coreg done now working on BA areas")
        vol_ba()
    else:
        print("Issue with coreg")
