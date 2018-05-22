import mne
from mne import create_info, EpochsArray
from nilearn.image import coord_transform
import pandas as pd
import pickle
import numpy as np
from mne.beamformer import make_lcmv, apply_lcmv
import random

affine = np.zeros((4, 4))
affine[0] = [0.88, 0, 0, -0.8]
affine[1] = [0, 0.97, 0, -3.32]
affine[2] = [0, 0.05, 0.88, -0.44]
affine[3] = [0, 0, 0, 1]

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

with open('/home/hpc_gagand87/beamformer/tal_to_BA.pkl', 'rb') as handle:
    tal_to_ba = pickle.load(handle)

data = pd.read_pickle('/home/hpc_gagand87/beamformer/data/dataset/dataset_sub1.pkl')
data  = data[data['area'] != 'NA']
data = data.reset_index(drop=True)
X = data.loc[:,data.columns != 'area']
y = data['area']
print(X.shape,y.shape)
ypred = []

for i in range(X.shape[0]):
    signal = X.loc[i,:]
    signal = np.reshape(signal, (43, 1000))
    yi = y[i]
    info = create_info(ch_names=['CH' + str(i) for i in range(1, 44)], sfreq=1000,
                       ch_types=['eeg' for i in range(1, 44)])
    raw = mne.io.RawArray(signal, info)
    raw_no_ref, _ = mne.set_eeg_reference(raw, ref_channels='average')
    signal = signal.reshape(1, 43, 1000)
    epochs = EpochsArray(signal, info)
    e_ref, _ = mne.set_eeg_reference(epochs, ref_channels='average')
    evoked = e_ref.average()

    fwd = mne.read_forward_solution('/home/hpc_gagand87/beamformer/fwd.fif')

    cov = mne.compute_covariance(e_ref, method='auto')
    filters = make_lcmv(evoked.info, fwd, cov, reg=0.05, pick_ori='max-power', weight_norm='unit-noise-gain',
                        reduce_rank=True)
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')

    max_vetrs_across_time = np.argmax(stc.rh_data,axis=0)
    verts = stc.vertices[0][max_vetrs_across_time]

    tbas = []
    for v in verts:
        mni = mne.vertex_to_mni(v, 0, 'fsaverage')[0]
        tal_coord = coord_transform(mni[0], mni[1], mni[2], affine)
        ba = tal_to_ba[int(tal_coord[0]), int(tal_coord[1]), int(tal_coord[2])]
        tbas.append(ba)
    if yi in tbas:
        ypred.append(yi)
    else:
        k = classes_to_keep.copy()
        k.remove(yi)
        ti = random.randint(0, 13)
        ypred.append(k[ti])
    print(ypred[i])
np.save('y.npy',y)
np.save('ypred.npy',ypred)

from sklearn.metrics import f1_score
print(f1_score(y, ypred, average='macro'))