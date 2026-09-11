"""Preserve person probability mass while removing within-person color emphasis."""
import numpy as np
from skin_color_sampling import sampling_distribution as base_distribution,draw_indices

ARMS=('person_mass','within_person_shuffle')


def sampling_distribution(data,arm,seed):
    if arm not in ARMS:raise ValueError('Unknown mass control')
    q,w,info=base_distribution(data,'color')
    sites,first,inverse,counts=np.unique(data['site'],return_index=True,return_inverse=True,return_counts=True)
    owner=data['patient'][first];mass=info['site_probability'].copy();rng=np.random.default_rng(seed+314159)
    for person in np.unique(owner):
        ix=np.flatnonzero(owner==person)
        mass[ix]=mass[ix].mean() if arm=='person_mass' else rng.permutation(mass[ix])
    q=mass[inverse]/counts[inverse]
    return q,np.ones(len(q)),info|{'site_probability':mass}
