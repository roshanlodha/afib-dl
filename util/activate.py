#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import predict

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_vandy(af):
    af['la_any_modality'] = af['la_any_modality'].replace(".", None).astype('float')
    af = af[af['la_any_modality'].notna()]
    gaf = af.groupby('recurrence')
    af = gaf.apply(lambda x: x.sample(gaf.size().min()).reset_index(drop=True))
    return af

def add_activations(vandy):
    vandy['nifti_dir'] = '/home/lodhar/afib-dl/nifti/vandy/' + vandy['study_id'].astype(str) + '.nii.gz'
    vandy_with_scans = vandy[vandy['nifti_dir'].apply(lambda x: os.path.isfile(x))].reset_index(drop = True)
    vandy_with_scans['activations'] = vandy_with_scans.apply(lambda row: predict.predict_activations(row['nifti_dir']), axis=1)
    flattened_activations = np.array(vandy_with_scans['activations'].tolist()).reshape(-1, 512)
    activations_df = pd.DataFrame(flattened_activations, columns=[f'activation_{i}' for i in range(512)])
    activated_vandy = pd.concat([vandy_with_scans.drop(['activations'], axis = 1), activations_df], axis=1)
    return activated_vandy

vandy = pd.read_csv('/home/lodhar/afib-dl/data/vanderbilt_ct_phenotype_2-14-23.csv')
vandy = load_and_clean_vandy(vandy)
vandy = add_activations(vandy)
vandy.to_csv('/home/lodhar/afib-dl/data/activated_vandy.csv')