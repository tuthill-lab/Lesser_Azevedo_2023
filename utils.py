#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 17:15:14 2020

@author: tony
"""

# create color map
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import requests

if __name__ == '__main__':
    pass


def xsegment(ct):
    if '_in_' not in ct:
        seg,rank = xltm(ct)
        if seg is None:
            seg,rank,miller = xmiller(ct)
            if seg is not None:
                return seg,rank
            else:
                seg = 'unknown'
                rank = 0
                return seg,rank
        else:
            return seg,rank
    else:
        seg = ct[ct.find('_in_')+4:]
        if '_' in seg:
            rank = seg[seg.find('_')+1:]
            seg = seg[0:seg.find('_')]
            if 'miller' in rank:
                rank = 0
        else:
            rank = 0
        return seg,rank
    
def xmiller(ct):
    if 'miller_' in ct:
        seg = 'thorax'
        rank = 0
        miller = ct[ct.find('miller_'):]
        return seg,rank, miller
    else:
        return None, None, None
        
def xltm(ct):
    if 'ltm' in ct:
        seg = 'ltm'
        if '_no_medial' in ct:
            rank = 'tiny'
            return seg,rank
        elif 'small' in ct:
            rank = 'small'
            return seg,rank
        else:
            rank = 'large'
            return seg,rank

    else: 
        return None,None
    
def xmuscle(ct):
    if '_in_' in ct:
        mscl = ct[0:ct.find('_in_')]
        if '_miller_' in ct:
            mscl = mscl + ct[ct.find('_miller_'):]
    elif 'ltm_' in ct:
        mscl = 'ltm'
    else:
        mscl = 'unknown'
        
    return mscl
    
def xnerve(clss):
    nerve = [clss[0]]
    
    if 'A' in nerve:
        return 'Accessory'
    if 'L' in nerve:
        return 'Leg'
    if 'D' in nerve:
        return 'Dorsal'
    if 'V' in nerve:
        return 'Ventral'
    
    fcn = 'unknown'
    return fcn
   
    
def xfunction(ct):
    if 'miller_32' in ct:
        return 'stance'
    if 'miller_29' in ct:
        return 'stance'
    if 'miller_28_30' in ct:
        return 'swing'
    if 'miller_31' in ct:
        return 'swing'
    if '_33' in ct: # 'miller_33' in ct:
        return 'swing'
    if 'flex' in ct:
        return 'flex'
    if 'exten' in ct:
        return 'extend'
    if 'tarsus_depressor' in ct:
        return 'flex'
    if 'depressor' in ct:
        return 'extend'
    if 'tergotrochanter' in ct:
        return 'extend'
    if 'reductor' in ct:
        return 'reductor'
    if 'ltm' in ct:
        return 'claw'
    if 'promotor' in ct:
        return 'flex'
    
    fcn = 'unknown'
    return fcn
   

def multiindex_include(mi,keep):
    # keep all tuples in the multiindex, mi, that contain keep
    lidx = []
    for indx in mi:
        for a in keep:
            if a in indx:
                
                ok = True
            else:
                ok = False
                break
        if ok:
            lidx = lidx+[True]
        else:
            lidx = lidx+[False]
    larr = np.array(lidx)
    mi_out = mi[larr];
    return mi_out

def multiindex_exclude(mi,non):
    # exclude all tuples in the multiindex, mi, that contain non
    lidx = []
    for indx in mi:
        if not non in indx:
            lidx = lidx+[False]
        else:
            lidx = lidx+[True]
    larr = np.array(lidx)
    larr = np.invert(larr)
    mi_out = mi[larr];
    return mi_out
    

# Given a dataframe of inputs, df, and a lower bound on the number of connections to keep
# Get a dataframe with counts of pre/post connections .
def group_and_count_inputs(df, thresh = 5):

    # count the number of synapses between pairs of pre and post synaptic inputs
    syn_in_conn=df.groupby(['pre_pt_root_id','post_pt_root_id']).transform(len)['id']
    # save this result in a new column and reorder the index
    df['syn_in_conn']=syn_in_conn
    df = df[['id', 'pre_pt_root_id','post_pt_root_id','score','syn_in_conn','has_soma','sensory','neck']].sort_values('syn_in_conn', ascending=False).reset_index()

    # Filter out small synapses between pairs of neurons and now print the shape
    df = df[df['syn_in_conn']>=thresh]
    # print(df.shape)
    return df

# Given a data frame of pre/post pairs, with a column  grouped connections 
def create_pre_post_df(mn_inputs_df,mi):
    # Find the unique premotor neurons
    pre_mn_segIDs, idx = np.unique(mn_inputs_df['pre_pt_root_id'].to_list(),return_index=True)

    # Find out if they have somas
    has_soma = mn_inputs_df['has_soma'][idx]
 
    # Find out if they are sensory
    sensory = mn_inputs_df['sensory'][idx]

    # Find out if they go through the neck connective
    neck = mn_inputs_df['neck'][idx]

    # Placeholder to find out if a neuron is local to T1
    local = neck
    local.loc[:] = False

    # Create multi index out of key info
    arrays = [
        pre_mn_segIDs,
        has_soma,
        sensory,
        neck,
        local
            ]
    pmn_tuples = list(zip(*arrays))
    # pmn_index = pd.MultiIndex.from_tuples(pmn_tuples, names=['segID','has_soma','sensory','descending','ascending','intersegmental','local'])
    pmn_index = pd.MultiIndex.from_tuples(pmn_tuples, names=['segID','has_soma','sensory','neck','local'])

    # Create a data frame with premotor neurons along the rows (index) and motor neurons along the columns
    if isinstance(mi,int):
        mi = [mi]
        
    dfdata = np.zeros([len(pre_mn_segIDs),len(mi)],dtype=int)
    mn_df = pd.DataFrame(dfdata,index=pmn_index, columns=mi)

    # Simplify the column indices
    mn_df.columns = mn_df.columns.get_level_values('segID')
    for index, row in mn_inputs_df.iterrows():
        mn_df.at[row.pre_pt_root_id,row.post_pt_root_id] = row.syn_in_conn

    # put the multiindex back
    mn_df.columns = mi
    return mn_df

def ordered_list_of_premotor_segIDs(client,segID,thresh = 5):
    a = [segID]
    arrays = [
        a]
    segID_tuples = list(zip(*arrays))
    segID_mi = pd.MultiIndex.from_tuples(segID_tuples, names=['segID'])
    mn_inputs_df = client.materialize.synapse_query(post_ids = segID_mi.get_level_values('segID').to_list())
    mn_inputs_df = group_and_count_inputs(mn_inputs_df, thresh = 5)
    partner_df = create_pre_post_df(mn_inputs_df,segID_mi)
    return partner_df

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    dend_dict = dendrogram(linkage_matrix, **kwargs)
    
    # sorted order of indices found through clustering
    clustered_order = dend_dict['ivl']
    plt.close()
    
    return clustered_order


# def get_segment_fcn_order(**kwargs):
#     reorder_L_bysegfcn = [
#         6,7,8,9,         # thorax swing miller 32
#         10,11,            # thorax swing miller 29
#         67, 68,          # thorax stance miller 31 
#         5,              # thorax stance Miller 33 
#         12,13,14,15,      # thorax stance miller 28 30
#         61,62,63,64,     # thorax/coxa tergotrochanter
#         65, 66,          # thorax/coxa extracoxal trochanter extensor
#         16, 17,          # coxa extensors
#         1, 0, 2, 3, 4,   # coxa flexors Acc
#         58,59,60,        # coxa flexors Ven
#         18,19,20,        # coxa troch reductor/promotor
#         52,53,54,55,56,57,  # femur reductor in the coxa
#         21,22,           # femur Feti, Seti
#         32,33,34,35,36,  # femur main tibia flexors
#         28,29,30,31,     # femur, aux tibia flexors B
#         23,24,           # femur, aux tibia flexors A1
#         25,26,27,        # femur, aux tibia flexors A2
#         37,              # femur, aux tibia flexors E
#         44, 45,          # femur, ltm tiny
#         42, 43,          # femur/tibia, ltm small
#         38, 39, 40, 41,  # femur/tibia, ltm large
#         46, 47,          # tibia, particular, ventral u, medial
#         50,              # tibia, B
#         48, 49,          # tibia, tarsal depressors A
#         51,              # tibia E
#          ]
    
#     reorder_L_bysegfcn = np.array(reorder_L_bysegfcn)
#     makespace = reorder_L_bysegfcn>51
#     reorder_R_bysegfcn = reorder_L_bysegfcn + makespace.astype(int)
#     reorder_R_bysegfcn = reorder_R_bysegfcn+69
#     reorder_L_bysegfcn = reorder_L_bysegfcn.tolist()
#     reorder_R_bysegfcn = reorder_R_bysegfcn.tolist() + [121]
#     reorder_bysegfcn = reorder_L_bysegfcn + reorder_R_bysegfcn
    
#     if 'side' in kwargs.keys():
#         if kwargs['side'] == 'left':
#             print(kwargs)
#             return reorder_L_bysegfcn
#         elif kwargs['side'] == 'right':
#             print(kwargs)
#             return reorder_R_bysegfcn
#     return reorder_bysegfcn


def get_motor_pool_tuple_dict():
    All = slice(None)
    motor_pool_tuple_dict = {
        'thorax_stance':                                             ('L','Accessory', 'thorax', 'stance', ['sternal_posterior_rotator_miller_32','pleural_remotor_and_abductor_miller_29'], All,All),
        'sternal_posterior_rotator_miller_32':                      ('L','Accessory', 'thorax', 'stance', 'sternal_posterior_rotator_miller_32', All,All),
        'pleural_remotor_and_abductor_miller_29':                   ('L','Accessory', 'thorax', 'stance', 'pleural_remotor_and_abductor_miller_29', All,All),
        
        'thorax_swing':                                            ('L',All, 'thorax', 'swing', ['tergopleural_promotor_pleural_promotor_miller_28_30','sternal_anterior_rotator_miller_31','sternal_adductor_miller_miller_in_thorax_miller_33'], All,All),
        'sternal_anterior_rotator_miller_31':                       ('L','Ventral', 'thorax', 'swing', 'sternal_anterior_rotator_miller_31', All,All),
        'sternal_adductor_miller_miller_in_thorax_miller_33':       ('L','Accessory', 'thorax', 'swing', 'sternal_adductor_miller_miller_in_thorax_miller_33', All,All),
        'tergopleural_promotor_pleural_promotor_miller_28_30':      ('L', 'Dorsal', 'thorax', 'swing', 'tergopleural_promotor_pleural_promotor_miller_28_30', All,All),
        
        'trochanter_extension':                 ('L',All, ['thorax','coxa'], 'extend', ['tergotrochanter','extracoxal_trochanter_depressor','trochanter_extensor'], All,All),
        'tergotrochanter':                      ('L', 'Ventral', 'thorax', 'extend', 'tergotrochanter', All, All),
        'extracoxal_trochanter_depressor':      ('L', 'Ventral', 'thorax', 'extend', 'extracoxal_trochanter_depressor', 'Vn',All),
        'trochanter_extensor':                  ('L', 'Leg', 'coxa', 'extend', 'trochanter_extensor', 'Lg', All),
        
        'trochanter_flexion':                   ('L', All, 'coxa', 'flex', ['trochanter_flexor','trochanter_promotor'], All, All),
        'trochanter_flexor_muscle':             ('L', All, 'coxa', 'flex', 'trochanter_flexor', ['Ac','Vn'], All),
        'trochanter_flexor_acc':                ('L', 'Accessory', 'coxa', 'flex', 'trochanter_flexor', ['AcPst','Ac',], All),
        'trochanter_flexor_ven':                ('L', 'Ventral', 'coxa', 'flex', 'trochanter_flexor', 'Vn', All),
        'trochanter_promotor':                  ('L', 'Leg', 'coxa', 'flex', 'trochanter_promotor', 0, All),
        
        'femur_reductor':                       ('L', 'Leg', 'trochanter', 'reductor', 'femur_reductor', All, All),
        
        'tibia_extensor':                       ('L', 'Leg', 'femur', 'extend', 'tibia_extensor', ['seti','feti'], All),
        'tibia_extensor_flexorA':               ('L', 'Leg', 'femur', ['extend','flex'], ['tibia_extensor','auxiliary_tibia_flexor','main_tibia_flexor'], ['seti','feti','Bslow','0','1','2','3','4'], All),
        'tibia_extensor_mainflexor_auxA':       ('L', 'Leg', 'femur', ['extend','flex'], ['tibia_extensor','main_tibia_flexor','auxiliary_tibia_flexor'], ['seti','feti','0','1','2','3','4','Bslow'], All),
        'main_tibia_flexor':                    ('L', 'Leg', 'femur', 'flex', ['main_tibia_flexor','auxiliary_tibia_flexor'], ['0','1','2','3','4','Bslow'], All),  #('L', 'Leg', 'femur', 'flex', 'main_tibia_flexor', All, All),
        'aux_main_tibia_flexor':                ('L', 'Leg', 'femur', 'flex', ['auxiliary_tibia_flexor','main_tibia_flexor'], ['Bslow','0','1','2','3','4'], All),  #('L', 'Leg', 'femur', 'flex', 'main_tibia_flexor', All, All),
        'main_tibia_flexor_muscle':             ('L', 'Leg', 'femur', 'flex', ['main_tibia_flexor'], All, All), 
        'auxiliary_tibia_flexor_A':             ('L', 'Leg', 'femur', 'flex', 'auxiliary_tibia_flexor', 'Bslow', All),
        'auxiliary_tibia_flexor_B':             ('L', 'Leg', 'femur', 'flex', 'auxiliary_tibia_flexor', ['A1','A2'], All),
        'auxiliary_tibia_flexor_E':             ('L', 'Leg', 'femur', 'flex', 'auxiliary_tibia_flexor', ['E'], All),
        'flexors_mft_B_E':                      ('L', 'Leg', 'femur', 'flex', ['main_tibia_flexor','auxiliary_tibia_flexor'], ['0','1','2','3','4','A1','A2','E'], All),  #('L', 'Leg', 'femur', 'flex', 'main_tibia_flexor', All, All),
        'extensors_and_atfE':                   ('L', 'Leg', 'femur', ['extend','flex'], ['tibia_extensor','auxiliary_tibia_flexor'], ['seti','feti','E'], All),

        'main_tibia_flexor_wtarsus':                    ('L', 'Leg', ['femur','tibia'], ['flex','unknown'], ['main_tibia_flexor','auxiliary_tibia_flexor', 'tarsus_unidentified'], ['0','1','2','3','4','Bslow'], All),  #('L', 'Leg', 'femur', 'flex', 'main_tibia_flexor', All, All),
        'aux_main_tibia_flexor_wtarsus':                ('L', 'Leg', ['femur','tibia'], ['flex','unknown'], ['auxiliary_tibia_flexor', 'main_tibia_flexor','tarsus_unidentified'], ['0','1','2','3','4','Bslow'], All),  #('L', 'Leg', 'femur', 'flex', 'main_tibia_flexor', All, All),
        'auxiliary_tibia_flexor_B_wtarsus':             ('L', 'Leg', ['femur','tibia'], ['flex','unknown'], ['auxiliary_tibia_flexor','tarsus_unidentified'], ['A1','A2'], All),
        'auxiliary_tibia_flexor_E_wtarsus':             ('L', 'Leg', ['femur','tibia'], ['flex','unknown'], ['auxiliary_tibia_flexor','tarsus_unidentified'], ['E'], All),

        'ltm':                                  ('L', 'Leg', 'ltm', 'claw', 'ltm', ['tiny','small','large'], All),
        'ltm_tiny':                             ('L', 'Leg', 'ltm', 'claw', 'ltm', ['tiny'], All),
        'ltm_small':                            ('L', 'Leg', 'ltm', 'claw', 'ltm', ['small'], All),
        'ltm_tiny_small':                       ('L', 'Leg', 'ltm', 'claw', 'ltm', ['tiny','small'], All),
        'ltm_large':                            ('L', 'Leg', 'ltm', 'claw', 'ltm', ['large'], All),
        'ltm_tibia':                            ('L', 'Leg', 'ltm', 'claw', 'ltm', ['large'], [648518346459661316,648518346477891821]),
        'ltm_femur':                            ('L', 'Leg', 'ltm', 'claw', 'ltm', ['large'], [648518346494032139,648518346479515840]),

        'tarsus_depressor':                     ('L', 'Leg', 'tibia', ['flex','unknown'], ['tarsus_depressor','tarsus_unidentified'], ['medial','ventralU','Bslow','A1','A2','E'], All),
        'tarsus_depressor_med_venU':            ('L', 'Leg', 'tibia', 'flex', 'tarsus_depressor', ['medial','ventralU'], All),
        'tarsus_depressor_medial':              ('L', 'Leg', 'tibia', 'flex', 'tarsus_depressor', ['medial'], All),
        'tarsus_depressor_ventralU':            ('L', 'Leg', 'tibia', 'flex', 'tarsus_depressor', ['ventralU'], All),
        'tarsus_depressor_noid':                ('L', 'Leg', 'tibia', 'unknown', 'tarsus_unidentified', ['Bslow','A1','A2','E'], All),       
        'tarsus_depressor_noid_A':              ('L', 'Leg', 'tibia', 'unknown', 'tarsus_unidentified', ['Bslow',], All),       
        'tarsus_depressor_noid_B':              ('L', 'Leg', 'tibia', 'unknown', 'tarsus_unidentified', ['A1','A2',], All),
        'tarsus_depressor_noid_B1':             ('L', 'Leg', 'tibia', 'unknown', 'tarsus_unidentified', ['A1','A2',], All),        
        'tarsus_depressor_noid_B2':             ('L', 'Leg', 'tibia', 'unknown', 'tarsus_unidentified', ['A2',], All),        
        'tarsus_depressor_noid_E':              ('L', 'Leg', 'tibia', 'unknown', 'tarsus_unidentified', ['E'], All),       
    }

    return motor_pool_tuple_dict

def get_right_motor_pool_tuple_dict():
    r_muscle_tuple_dict = get_motor_pool_tuple_dict()
    for key in r_muscle_tuple_dict.keys():
        val = r_muscle_tuple_dict[key]
        listA = list(val)
        listA[0] = 'R'
        r_muscle_tuple_dict[key] = tuple(listA)
        
    return r_muscle_tuple_dict


def sort_segment_fcn_index(col_idx,muscle_tuple_dict =get_motor_pool_tuple_dict()):
    # If you sort the index according to number of inputs first, then run this code, it should order the thing by seg and fcn, then by inputs
    if type(col_idx) != pd.core.indexes.multi.MultiIndex:
        print('col_idx must be a multiindex')
        raise TypeError

    col_idx = col_idx.to_frame()

    # thorax swing miller 28 30
    new_idx = col_idx.loc[
        muscle_tuple_dict['tergopleural_promotor_pleural_promotor_miller_28_30']
    ]

    # thorax swing miller 31 
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['sternal_anterior_rotator_miller_31']
    ]])

    # thorax swing Miller 33, only 1
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['sternal_adductor_miller_miller_in_thorax_miller_33']
    ]])
    
    
    # thorax stance miller 29
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['pleural_remotor_and_abductor_miller_29']
        ]])

     # thorax stance miller 32
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['sternal_posterior_rotator_miller_32']
    ]])
   

    # thorax/coxa tergotrochanter
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['tergotrochanter']
    ]])

    # thorax/coxa extracoxal trochanter extensor
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['extracoxal_trochanter_depressor']
    ]])

    # coxa extensors
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['trochanter_extensor']
    ]])

    # coxa flexors Acc Posterior cell bodies
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['trochanter_flexor_acc']
    ]])
    
    # coxa flexors Ven
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['trochanter_flexor_ven']
    ]])

    # coxa troch reductor/promotor
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['trochanter_promotor']
    ]])
 
    # femur reductor in the coxa
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['femur_reductor']
    ]])

    # femur Feti, Seti
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['tibia_extensor']
    ]])

    # femur main tibia flexors
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['main_tibia_flexor_muscle']
    ]])

     # femur, aux tibia flexors B
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['auxiliary_tibia_flexor_A']
    ]])

    # femur, aux tibia flexors A
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['auxiliary_tibia_flexor_B']
    ]])
    
    # femur, aux tibia flexors E
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['auxiliary_tibia_flexor_E']
    ]])
    
    # ltm 
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['ltm']
    ]])

    # tibia, particular, ventral u, medial
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['tarsus_depressor_med_venU']
    ]])

    # tibia, B
    new_idx = pd.concat([new_idx,col_idx.loc[
        muscle_tuple_dict['tarsus_depressor_noid']
    ]])

    idx = pd.MultiIndex.from_frame(new_idx)
    return idx

def save_df_csv(temp_df, name):
    today = date.today()
    d1 = today.strftime("%Y%m%d")
    fn = './dfs_saved/' + name + '_' + d1 + '.csv'
    temp_df.to_csv(fn)
    print(fn)
    print(temp_df.shape)

def total_sort_save_df_csv(temp_df, name=''):
    total = temp_df.sum(axis=1)
    idx = temp_df.index.to_frame()
    idx['total_synapses'] = total
    newidx = pd.MultiIndex.from_frame(idx)
    temp_df.index = newidx
    temp_df = temp_df.sort_index(axis=0, level='total_synapses',sort_remaining=False,ascending=False)
    if name != '':
        save_df_csv(temp_df, name)
    return temp_df

def save_pre_to_mn_df_csv(temp_df):
    name = 'pre_to_mn_df_csv'
    today = date.today()
    d1 = today.strftime("%Y%m%d")
    fn = './dfs_pre_to_mn/' + name + '_' + d1 + '.csv'
    temp_df.to_csv(fn)
    print(fn)
    print(temp_df.shape)

def save_df_as_pickle(temp_df,name=''):
    today = date.today()
    d1 = today.strftime("%Y%m%d")
    fn = './dfs_saved/' + name + '_' + d1 + '.pkl'
    temp_df.to_pickle(fn)
    print(fn)
    print(temp_df.shape)

def xticks_from_mnmi(mn_mi):
    mn_msc = mn_mi['muscle'].to_list()
    mn_rnk = mn_mi['rank'].to_list()
    x =[]
    cnt = -1
    curmsc = ''

    for msc, rnk in zip(mn_msc, mn_rnk):
        if msc is not curmsc:
            cnt = cnt+1
            currnk = rnk
            
        if msc=='auxiliary_tibia_flexor':
            if rnk is not currnk:
                cnt = cnt+1
                currnk = rnk
                # print(msc, currnk)
        x = x+[cnt]
        cnt = cnt+1
        curmsc = msc

    return x

def xticks_from_pools(mn_mi):
    pool_keys = [
        'thorax_swing',
        'thorax_stance',
        'trochanter_extension',
        'trochanter_flexion',
        'femur_reductor',
        'tibia_extensor',
        'main_tibia_flexor',
        # 'auxiliary_tibia_flexor_A',
        'auxiliary_tibia_flexor_B',
        'auxiliary_tibia_flexor_E',
        'ltm',
        'tarsus_depressor'
        ]
    muscle_tuple_dict = get_motor_pool_tuple_dict()
    x =[]
    cnt = 0
    curkey = ''

    for pk in pool_keys:
        x = x + [cnt + i for i in range(mn_mi.loc[muscle_tuple_dict[pk]].shape[0])]
        cnt = x[-1] + 2

    return x

def mn_labels(mn_mi,depth=None):
    mn_seg = mn_mi['segment'].to_list()
    mn_fcn = mn_mi['function'].to_list()
    mn_msc = mn_mi['muscle'].to_list()
    idx = [str(i) for i in range(len(mn_msc))]

    if depth is None:
        lbls = [i + '_' + j + '_' + k + '_' + l for i, j, k,l in zip(mn_seg, mn_fcn, mn_msc, idx,)]
    elif depth=='rank':
        mn_rnk = mn_mi['rank'].to_list()
        rnk = [str(i) for i in mn_rnk]
        lbls = [i + '_' + j + '_' + k + '_' + l + '_' + m for i, j, k,l,m in zip(mn_seg, mn_fcn, mn_msc, rnk, idx,)]
    return lbls

def segIDs_from_pts_service(pts, cv,
                         service_url = 'https://services.itanna.io/app/transform-service/query/dataset/fanc_v4/s/{}/values_array_string_response/',
                         scale = 2,
                         return_roots=True):
        print('Getting segIDs')
        #Reshape from list entries if dataframe column is passed
        if len(pts.shape) == 1:
            pts = np.concatenate(pts).reshape(-1,3)

        service_url = service_url.format(scale)
        pts = np.array(pts, dtype=np.uint32)
        ndims = len(pts.shape)
        if ndims == 1:
            pts = pts.reshape([-1,3])
        r = requests.post(service_url, json={
            'x': list(pts[:, 0].astype(str)),
            'y': list(pts[:, 1].astype(str)),
            'z': list(pts[:, 2].astype(str))
        })
        try:
            r = r.json()['values'][0]
            sv_ids = [int(i) for i in r]
            print('using .json')
            if return_roots is True:
                return cv.get_roots(sv_ids)
            else:
                return sv_ids
        except:
            print('not using .json')
            return r


def desc_sens_local_etc_cmap():
    colors = ["#95d0fc","#90e4c1","#916e99","#069af3","#d8dcd6"]
    cmap = sns.set_palette(sns.color_palette(colors))
    return cmap

def make_json(seg_ids=None, hidden_ids=None,name="published FANC neurons"):
    json_str = {
        "layers": [
            {
            "source": "precomputed://gs://zetta_lee_fly_vnc_001_precomputed/fanc_v4_em",
            "type": "image",
            "blend": "default",
            "shaderControls": {},
            "name": "FANCv4"
            },
            {
            "source": "graphene://https://cave.fanc-fly.com/segmentation/table/mar2021_prod",
            "type": "segmentation_with_graph",
            "colorSeed": 1792288153,
            "segments": seg_ids,
            "skeletonRendering": {
                "mode2d": "lines_and_points",
                "mode3d": "lines"
            },
            "graphOperationMarker": [
                {
                "annotations": [],
                "tags": []
                },
                {
                "annotations": [],
                "tags": []
                }
            ],
            "pathFinder": {
                "color": "#ffff00",
                "pathObject": {
                "annotationPath": {
                    "annotations": [],
                    "tags": []
                },
                "hasPath": False
                }
            },
            "name": name #"seg_Mar2021_proofreading"
            },
            {
            "type": "segmentation",
            "mesh": "precomputed://gs://zetta_lee_fly_vnc_001_precomputed/vnc1_full_v3align_2/brain_regions",
            "objectAlpha": 0.1,
            "hideSegmentZero": False,
            "ignoreSegmentInteractions": True,
            "segmentColors": {
                "1": "#bfbfbf",
                "2": "#d343d6"
            },
            "segments": [
                "1",
                "2"
            ],
            "skeletonRendering": {
                "mode2d": "lines_and_points",
                "mode3d": "lines"
            },
            "name": "volume outlines"
            }
        ],
        "navigation": {
            "pose": {
            "position": {
                "voxelSize": [
                4.300000190734863,
                4.300000190734863,
                45
                ],
                "voxelCoordinates": [
                    32476.95703125,
                    99454.96875,
                    1891.22607421875
                ]
            }
            },
            "zoomFactor": 14.975854772570637
        },
        "perspectiveOrientation": [
            0.0034316908568143845,
            -0.0034852433018386364,
            0.0022593876346945763,
            0.9999854564666748
        ],
        "perspectiveZoom": 4317,
        "showSlices": False,
        "gpuMemoryLimit": 4000000000,
        "systemMemoryLimit": 4000000000,
        "concurrentDownloads": 64,
        "jsonStateServer": "https://global.daf-apis.com/nglstate/api/v1/post",
        "selectedLayer": {
            "layer": "seg_Mar2021_proofreading",
            "visible": True
        },
        "layout": "xy-3d"
    }
    return json_str

# {
#   "layers": [
#     {
#       "source": "precomputed://gs://zetta_lee_fly_vnc_001_precomputed/fanc_v4_em",
#       "type": "image",
#       "blend": "default",
#       "shaderControls": {},
#       "name": "FANCv4"
#     },
#     {
#       "source": "graphene://https://cave.fanc-fly.com/segmentation/table/mar2021_prod",
#       "type": "segmentation_with_graph",
#       "colorSeed": 1792288153,
#       "segmentColors": {
#         "648518346484097831": "#ff3737",
#         "648518346494281192": "#28e1ff",
#         "648518346486744291": "#7dff7d",
#         "648518346484474627": "#7dff7d",
#         "648518346506877256": "#2dfff5",
#         "648518346490510589": "#7d7dff",
#         "648518346496552962": "#37c337",
#         "648518346493943072": "#7dff7d",
#         "648518346497547510": "#0a9b0a",
#         "648518346499753100": "#1effff",
#         "648518346473993325": "#ff7d7d",
#         "648518346486525133": "#ffa01e",
#         "648518346496651309": "#ff7d7d",
#         "648518346494755506": "#ffb40a",
#         "648518346498254576": "#1fe0f9",
#         "648518346495911014": "#37c337",
#         "648518346493717646": "#19d4f5",
#         "648518346483133092": "#32e6ff",
#         "648518346497452278": "#19dff5",
#         "648518346513752089": "#ff9100",
#         "648518346486075538": "#0a9b0a",
#         "648518346493057308": "#ffc832",
#         "648518346502925179": "#0a9b0a",
#         "648518346520152913": "#37ff37",
#         "648518346477819093": "#7d7dff",
#         "648518346488315887": "#0ff5d7",
#         "648518346493664248": "#7dff7d"
#       },
#       "segments": [
#         "648518346487932496"
#       ],
#       "skeletonRendering": {
#         "mode2d": "lines_and_points",
#         "mode3d": "lines"
#       },
#       "graphOperationMarker": [
#         {
#           "annotations": [],
#           "tags": []
#         },
#         {
#           "annotations": [],
#           "tags": []
#         }
#       ],
#       "pathFinder": {
#         "color": "#ffff00",
#         "pathObject": {
#           "annotationPath": {
#             "annotations": [],
#             "tags": []
#           },
#           "hasPath": false
#         }
#       },
#       "name": "seg_Mar2021_proofreading"
#     },
#     {
#       "source": "precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/synapses/postsynapses_May2021",
#       "type": "image",
#       "blend": "default",
#       "shader": "void main() { emitRGBA(vec4(1, 0, 1, toNormalized(getDataValue()))); }",
#       "shaderControls": {},
#       "name": "synapses_May2021",
#       "visible": false
#     },
#     {
#       "type": "segmentation",
#       "mesh": "precomputed://gs://zetta_lee_fly_vnc_001_precomputed/vnc1_full_v3align_2/brain_regions",
#       "objectAlpha": 0.1,
#       "hideSegmentZero": false,
#       "ignoreSegmentInteractions": true,
#       "segmentColors": {
#         "1": "#bfbfbf",
#         "2": "#d343d6"
#       },
#       "segments": [
#         "1",
#         "2"
#       ],
#       "skeletonRendering": {
#         "mode2d": "lines_and_points",
#         "mode3d": "lines"
#       },
#       "name": "volume outlines"
#     }
#   ],
#   "navigation": {
#     "pose": {
#       "position": {
#         "voxelSize": [
#           4.300000190734863,
#           4.300000190734863,
#           45
#         ],
#         "voxelCoordinates": [
#           38488,
#           119558,
#           603
#         ]
#       }
#     },
#     "zoomFactor": 110.65743104396275
#   },
#   "showDefaultAnnotations": false,
#   "perspectiveZoom": 4317.621930475839,
#   "showSlices": false,
#   "gpuMemoryLimit": 4000000000,
#   "systemMemoryLimit": 4000000000,
#   "concurrentDownloads": 64,
#   "jsonStateServer": "https://global.daf-apis.com/nglstate/api/v1/post",
#   "selectedLayer": {
#     "layer": "seg_Mar2021_proofreading",
#     "visible": true
#   },
#   "layout": "3d"
# }

def black_parula():
    # 0.2422, 0.1504, 0.6603
    cm_data = [[0.0, 0.0, 0.0],
    [0.2444, 0.1534, 0.6728],
    [0.2464, 0.1569, 0.6847],
    [0.2484, 0.1607, 0.6961],
    [0.2503, 0.1648, 0.7071],
    [0.2522, 0.1689, 0.7179],
    [0.254, 0.1732, 0.7286],
    [0.2558, 0.1773, 0.7393],
    [0.2576, 0.1814, 0.7501],
    [0.2594, 0.1854, 0.761],
    [0.2611, 0.1893, 0.7719],
    [0.2628, 0.1932, 0.7828],
    [0.2645, 0.1972, 0.7937],
    [0.2661, 0.2011, 0.8043],
    [0.2676, 0.2052, 0.8148],
    [0.2691, 0.2094, 0.8249],
    [0.2704, 0.2138, 0.8346],
    [0.2717, 0.2184, 0.8439],
    [0.2729, 0.2231, 0.8528],
    [0.274, 0.228, 0.8612],
    [0.2749, 0.233, 0.8692],
    [0.2758, 0.2382, 0.8767],
    [0.2766, 0.2435, 0.884],
    [0.2774, 0.2489, 0.8908],
    [0.2781, 0.2543, 0.8973],
    [0.2788, 0.2598, 0.9035],
    [0.2794, 0.2653, 0.9094],
    [0.2798, 0.2708, 0.915],
    [0.2802, 0.2764, 0.9204],
    [0.2806, 0.2819, 0.9255],
    [0.2809, 0.2875, 0.9305],
    [0.2811, 0.293, 0.9352],
    [0.2813, 0.2985, 0.9397],
    [0.2814, 0.304, 0.9441],
    [0.2814, 0.3095, 0.9483],
    [0.2813, 0.315, 0.9524],
    [0.2811, 0.3204, 0.9563],
    [0.2809, 0.3259, 0.96],
    [0.2807, 0.3313, 0.9636],
    [0.2803, 0.3367, 0.967],
    [0.2798, 0.3421, 0.9702],
    [0.2791, 0.3475, 0.9733],
    [0.2784, 0.3529, 0.9763],
    [0.2776, 0.3583, 0.9791],
    [0.2766, 0.3638, 0.9817],
    [0.2754, 0.3693, 0.984],
    [0.2741, 0.3748, 0.9862],
    [0.2726, 0.3804, 0.9881],
    [0.271, 0.386, 0.9898],
    [0.2691, 0.3916, 0.9912],
    [0.267, 0.3973, 0.9924],
    [0.2647, 0.403, 0.9935],
    [0.2621, 0.4088, 0.9946],
    [0.2591, 0.4145, 0.9955],
    [0.2556, 0.4203, 0.9965],
    [0.2517, 0.4261, 0.9974],
    [0.2473, 0.4319, 0.9983],
    [0.2424, 0.4378, 0.9991],
    [0.2369, 0.4437, 0.9996],
    [0.2311, 0.4497, 0.9995],
    [0.225, 0.4559, 0.9985],
    [0.2189, 0.462, 0.9968],
    [0.2128, 0.4682, 0.9948],
    [0.2066, 0.4743, 0.9926],
    [0.2006, 0.4803, 0.9906],
    [0.195, 0.4861, 0.9887],
    [0.1903, 0.4919, 0.9867],
    [0.1869, 0.4975, 0.9844],
    [0.1847, 0.503, 0.9819],
    [0.1831, 0.5084, 0.9793],
    [0.1818, 0.5138, 0.9766],
    [0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9709],
    [0.1785, 0.5296, 0.9677],
    [0.1778, 0.5349, 0.9641],
    [0.1773, 0.5401, 0.9602],
    [0.1768, 0.5452, 0.956],
    [0.1764, 0.5504, 0.9516],
    [0.1755, 0.5554, 0.9473],
    [0.174, 0.5605, 0.9432],
    [0.1716, 0.5655, 0.9393],
    [0.1686, 0.5705, 0.9357],
    [0.1649, 0.5755, 0.9323],
    [0.161, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254],
    [0.154, 0.5902, 0.9218],
    [0.1513, 0.595, 0.9182],
    [0.1492, 0.5997, 0.9147],
    [0.1475, 0.6043, 0.9113],
    [0.1461, 0.6089, 0.908],
    [0.1446, 0.6135, 0.905],
    [0.1429, 0.618, 0.9022],
    [0.1408, 0.6226, 0.8998],
    [0.1383, 0.6272, 0.8975],
    [0.1354, 0.6317, 0.8953],
    [0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.891],
    [0.1253, 0.6453, 0.8887],
    [0.1219, 0.6497, 0.8862],
    [0.1185, 0.6541, 0.8834],
    [0.1152, 0.6584, 0.8804],
    [0.1119, 0.6627, 0.877],
    [0.1085, 0.6669, 0.8734],
    [0.1048, 0.671, 0.8695],
    [0.1009, 0.675, 0.8653],
    [0.0964, 0.6789, 0.8609],
    [0.0914, 0.6828, 0.8562],
    [0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462],
    [0.0713, 0.6938, 0.8409],
    [0.0628, 0.6972, 0.8355],
    [0.0535, 0.7006, 0.8299],
    [0.0433, 0.7039, 0.8242],
    [0.0328, 0.7071, 0.8183],
    [0.0234, 0.7103, 0.8124],
    [0.0155, 0.7133, 0.8064],
    [0.0091, 0.7163, 0.8003],
    [0.0046, 0.7192, 0.7941],
    [0.0019, 0.722, 0.7878],
    [0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752],
    [0.0046, 0.7301, 0.7688],
    [0.0094, 0.7327, 0.7623],
    [0.0162, 0.7352, 0.7558],
    [0.0253, 0.7376, 0.7492],
    [0.0369, 0.74, 0.7426],
    [0.0504, 0.7423, 0.7359],
    [0.0638, 0.7446, 0.7292],
    [0.077, 0.7468, 0.7224],
    [0.0899, 0.7489, 0.7156],
    [0.1023, 0.751, 0.7088],
    [0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.695],
    [0.1354, 0.7572, 0.6881],
    [0.1448, 0.7593, 0.6812],
    [0.1532, 0.7614, 0.6741],
    [0.1609, 0.7635, 0.6671],
    [0.1678, 0.7656, 0.6599],
    [0.1741, 0.7678, 0.6527],
    [0.1799, 0.7699, 0.6454],
    [0.1853, 0.7721, 0.6379],
    [0.1905, 0.7743, 0.6303],
    [0.1954, 0.7765, 0.6225],
    [0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065],
    [0.2118, 0.7828, 0.5983],
    [0.2178, 0.7849, 0.5899],
    [0.2244, 0.7869, 0.5813],
    [0.2318, 0.7887, 0.5725],
    [0.2401, 0.7905, 0.5636],
    [0.2491, 0.7922, 0.5546],
    [0.2589, 0.7937, 0.5454],
    [0.2695, 0.7951, 0.536],
    [0.2809, 0.7964, 0.5266],
    [0.2929, 0.7975, 0.517],
    [0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975],
    [0.3301, 0.8002, 0.4876],
    [0.3424, 0.8009, 0.4774],
    [0.3548, 0.8016, 0.4669],
    [0.3671, 0.8021, 0.4563],
    [0.3795, 0.8026, 0.4454],
    [0.3921, 0.8029, 0.4344],
    [0.405, 0.8031, 0.4233],
    [0.4184, 0.803, 0.4122],
    [0.4322, 0.8028, 0.4013],
    [0.4463, 0.8024, 0.3904],
    [0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691],
    [0.4899, 0.8002, 0.3586],
    [0.5044, 0.7993, 0.348],
    [0.5187, 0.7982, 0.3374],
    [0.5329, 0.797, 0.3267],
    [0.547, 0.7957, 0.3159],
    [0.5609, 0.7943, 0.305],
    [0.5748, 0.7929, 0.2941],
    [0.5886, 0.7913, 0.2833],
    [0.6024, 0.7896, 0.2726],
    [0.6161, 0.7878, 0.2622],
    [0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423],
    [0.6567, 0.7818, 0.2329],
    [0.6701, 0.7796, 0.2239],
    [0.6833, 0.7773, 0.2155],
    [0.6963, 0.775, 0.2075],
    [0.7091, 0.7727, 0.1998],
    [0.7218, 0.7703, 0.1924],
    [0.7344, 0.7679, 0.1852],
    [0.7468, 0.7654, 0.1782],
    [0.759, 0.7629, 0.1717],
    [0.771, 0.7604, 0.1658],
    [0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.157],
    [0.806, 0.7529, 0.1546],
    [0.8172, 0.7505, 0.1535],
    [0.8281, 0.7481, 0.1536],
    [0.8389, 0.7457, 0.1546],
    [0.8495, 0.7435, 0.1564],
    [0.86, 0.7413, 0.1587],
    [0.8703, 0.7392, 0.1615],
    [0.8804, 0.7372, 0.165],
    [0.8903, 0.7353, 0.1695],
    [0.9, 0.7336, 0.1749],
    [0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.189],
    [0.9272, 0.7298, 0.1973],
    [0.9357, 0.729, 0.2061],
    [0.944, 0.7285, 0.2151],
    [0.9523, 0.7284, 0.2237],
    [0.9606, 0.7285, 0.2312],
    [0.9689, 0.7292, 0.2373],
    [0.977, 0.7304, 0.2418],
    [0.9842, 0.733, 0.2446],
    [0.99, 0.7365, 0.2429],
    [0.9946, 0.7407, 0.2394],
    [0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309],
    [0.9972, 0.7569, 0.2267],
    [0.9971, 0.7626, 0.2224],
    [0.9969, 0.7683, 0.2181],
    [0.9966, 0.774, 0.2138],
    [0.9962, 0.7798, 0.2095],
    [0.9957, 0.7856, 0.2053],
    [0.9949, 0.7915, 0.2012],
    [0.9938, 0.7974, 0.1974],
    [0.9923, 0.8034, 0.1939],
    [0.9906, 0.8095, 0.1906],
    [0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846],
    [0.9835, 0.828, 0.1817],
    [0.9807, 0.8342, 0.1787],
    [0.9778, 0.8404, 0.1757],
    [0.9748, 0.8467, 0.1726],
    [0.972, 0.8529, 0.1695],
    [0.9694, 0.8591, 0.1665],
    [0.9671, 0.8654, 0.1636],
    [0.9651, 0.8716, 0.1608],
    [0.9634, 0.8778, 0.1582],
    [0.9619, 0.884, 0.1557],
    [0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507],
    [0.9596, 0.9023, 0.148],
    [0.9595, 0.9084, 0.145],
    [0.9597, 0.9143, 0.1418],
    [0.9601, 0.9203, 0.1382],
    [0.9608, 0.9262, 0.1344],
    [0.9618, 0.932, 0.1304],
    [0.9629, 0.9379, 0.1261],
    [0.9642, 0.9437, 0.1216],
    [0.9657, 0.9494, 0.1168],
    [0.9674, 0.9552, 0.1116],
    [0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001],
    [0.973, 0.9724, 0.0938],
    [0.9749, 0.9782, 0.0872],
    [0.9769, 0.9839, 0.0805]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    return parula_map

def white_ice():
    
    cmdata = [[1 , 1, 1],
       [0.9180593 , 0.99181354, 0.99283286],
       [0.91133005, 0.98744495, 0.98944426],
       [0.90450134, 0.9831399 , 0.98600058],
       [0.89759175, 0.9788908 , 0.98250515],
       [0.89061741, 0.97469071, 0.9789641 ],
       [0.88359326, 0.97053314, 0.97538393],
       [0.87652837, 0.96641348, 0.9717733 ],
       [0.86942981, 0.96232795, 0.96814019],
       [0.86230259, 0.95827357, 0.96449203],
       [0.85515095, 0.95424775, 0.96083521],
       [0.84798207, 0.95024718, 0.95717374],
       [0.84079256, 0.9462718 , 0.9535152 ],
       [0.83358579, 0.94231965, 0.94986338],
       [0.82636495, 0.9383889 , 0.94622154],
       [0.81912624, 0.93447982, 0.9425952 ],
       [0.81187625, 0.93058973, 0.93898531],
       [0.80460936, 0.9267195 , 0.93539728],
       [0.79733058, 0.92286698, 0.93183206],
       [0.79003624, 0.91903253, 0.92829379],
       [0.78272987, 0.91521448, 0.92478363],
       [0.77540864, 0.91141295, 0.92130508],
       [0.76807541, 0.9076265 , 0.91785934],
       [0.76072821, 0.90385503, 0.91444944],
       [0.7533689 , 0.90009739, 0.91107686],
       [0.74599695, 0.89635308, 0.90774403],
       [0.73861316, 0.89262124, 0.90445285],
       [0.7312183 , 0.88890104, 0.90120526],
       [0.72381291, 0.88519169, 0.89800335],
       [0.71639823, 0.88149221, 0.89484897],
       [0.70897547, 0.87780166, 0.89174404],
       [0.70154587, 0.87411906, 0.88869057],
       [0.69411182, 0.87044318, 0.88569014],
       [0.6866745 , 0.86677306, 0.88274497],
       [0.67923764, 0.86310717, 0.87985618],
       [0.67180273, 0.85944449, 0.87702608],
       [0.66437458, 0.85578326, 0.87425541],
       [0.65695573, 0.85212228, 0.87154615],
       [0.64955148, 0.84845972, 0.86889888],
       [0.64216625, 0.84479403, 0.86631468],
       [0.63480525, 0.84112351, 0.8637942 ],
       [0.62747522, 0.8374462 , 0.86133712],
       [0.62018101, 0.8337606 , 0.85894417],
       [0.6129312 , 0.83006454, 0.85661357],
       [0.60573106, 0.8263566 , 0.85434539],
       [0.59858923, 0.8226348 , 0.85213717],
       [0.59151241, 0.81889765, 0.84998727],
       [0.58450804, 0.81514366, 0.84789305],
       [0.57758278, 0.81137158, 0.84585184],
       [0.57074395, 0.80758021, 0.84385969],
       [0.56399647, 0.80376887, 0.84191391],
       [0.55734648, 0.79993683, 0.84000992],
       [0.55079789, 0.79608381, 0.83814427],
       [0.544354  , 0.79220975, 0.83631331],
       [0.5380184 , 0.78831467, 0.8345121 ],
       [0.53179094, 0.78439917, 0.83273894],
       [0.52567488, 0.78046351, 0.83098749],
       [0.51966727, 0.77650874, 0.82925804],
       [0.51377031, 0.77253541, 0.82754379],
       [0.5079798 , 0.76854472, 0.82584588],
       [0.50229586, 0.76453755, 0.82415903],
       [0.49671496, 0.76051503, 0.82248288],
       [0.49123401, 0.75647827, 0.82081642],
       [0.48585194, 0.75242831, 0.81915472],
       [0.480563  , 0.74836626, 0.81750143],
       [0.47536584, 0.74429316, 0.81585182],
       [0.47025671, 0.74021004, 0.81420586],
       [0.46523129, 0.7361178 , 0.81256547],
       [0.46028743, 0.73201742, 0.8109275 ],
       [0.45542205, 0.72790979, 0.80929102],
       [0.4506311 , 0.72379561, 0.8076587 ],
       [0.44591175, 0.71967561, 0.80603022],
       [0.44126199, 0.71555064, 0.80440251],
       [0.43667866, 0.7114213 , 0.802777  ],
       [0.43215888, 0.70728805, 0.80115515],
       [0.42770031, 0.70315144, 0.79953693],
       [0.42330076, 0.69901195, 0.79792237],
       [0.4189584 , 0.69487019, 0.79630958],
       [0.41467103, 0.69072647, 0.79469983],
       [0.41043672, 0.68658107, 0.79309427],
       [0.40625381, 0.68243429, 0.79149303],
       [0.40212077, 0.6782864 , 0.78989626],
       [0.39803619, 0.67413764, 0.78830412],
       [0.39399882, 0.66998823, 0.78671676],
       [0.39000748, 0.66583833, 0.78513436],
       [0.38606113, 0.66168821, 0.78355612],
       [0.38215883, 0.6575379 , 0.78198297],
       [0.37829979, 0.65338743, 0.78041541],
       [0.37448333, 0.64923687, 0.77885356],
       [0.37070887, 0.64508626, 0.77729756],
       [0.36697594, 0.64093562, 0.7757475 ],
       [0.36328417, 0.63678492, 0.77420346],
       [0.3596333 , 0.63263416, 0.77266552],
       [0.35602315, 0.62848328, 0.77113373],
       [0.35245366, 0.62433221, 0.76960811],
       [0.34892482, 0.62018088, 0.76808867],
       [0.34543677, 0.61602918, 0.76657539],
       [0.34198969, 0.61187699, 0.76506822],
       [0.33858388, 0.60772419, 0.76356711],
       [0.3352197 , 0.60357064, 0.76207194],
       [0.33189762, 0.59941617, 0.76058259],
       [0.32861818, 0.5952606 , 0.7590989 ],
       [0.325382  , 0.59110377, 0.75762068],
       [0.32218978, 0.58694546, 0.7561477 ],
       [0.31904231, 0.58278547, 0.7546797 ],
       [0.31594045, 0.57862358, 0.75321637],
       [0.31288513, 0.57445957, 0.75175739],
       [0.30987735, 0.57029319, 0.75030237],
       [0.3069182 , 0.5661242 , 0.74885089],
       [0.30400882, 0.56195234, 0.74740249],
       [0.30115042, 0.55777733, 0.74595664],
       [0.29834428, 0.55359893, 0.74451279],
       [0.29559174, 0.54941683, 0.74307033],
       [0.29289419, 0.54523076, 0.74162858],
       [0.29025309, 0.54104043, 0.74018682],
       [0.28766991, 0.53684554, 0.73874426],
       [0.28514622, 0.53264579, 0.73730005],
       [0.28268357, 0.52844088, 0.73585328],
       [0.2802836 , 0.52423051, 0.73440296],
       [0.27794793, 0.52001436, 0.73294804],
       [0.27567821, 0.51579214, 0.73148736],
       [0.27347612, 0.51156355, 0.73001971],
       [0.27134332, 0.50732827, 0.72854378],
       [0.26928146, 0.50308602, 0.72705818],
       [0.26729219, 0.49883651, 0.7255614 ],
       [0.26537711, 0.49457945, 0.72405184],
       [0.26353778, 0.49031459, 0.7225278 ],
       [0.2617757 , 0.48604166, 0.72098746],
       [0.26009232, 0.48176042, 0.71942887],
       [0.25848896, 0.47747067, 0.71784998],
       [0.25696687, 0.47317219, 0.71624858],
       [0.25552716, 0.46886482, 0.71462235],
       [0.25417082, 0.46454843, 0.71296882],
       [0.25289865, 0.4602229 , 0.71128535],
       [0.25171129, 0.45588817, 0.70956918],
       [0.25060917, 0.45154422, 0.70781736],
       [0.24959252, 0.44719108, 0.70602681],
       [0.2486613 , 0.44282884, 0.70419426],
       [0.24781523, 0.43845764, 0.70231628],
       [0.24705372, 0.43407771, 0.70038927],
       [0.24637589, 0.42968935, 0.69840946],
       [0.24578055, 0.42529292, 0.69637292],
       [0.24526613, 0.4208889 , 0.69427555],
       [0.24483074, 0.41647785, 0.6921131 ],
       [0.24447207, 0.41206045, 0.68988119],
       [0.24418744, 0.40763748, 0.68757528],
       [0.24397378, 0.40320985, 0.68519075],
       [0.24382759, 0.39877858, 0.68272286],
       [0.24374497, 0.39434483, 0.68016682],
       [0.24372159, 0.3899099 , 0.67751781],
       [0.24375272, 0.3854752 , 0.67477099],
       [0.24383334, 0.38104229, 0.6719216 ],
       [0.2439578 , 0.37661289, 0.66896489],
       [0.24412005, 0.37218889, 0.66589627],
       [0.24431385, 0.36777225, 0.66271137],
       [0.24453264, 0.36336505, 0.65940606],
       [0.24476962, 0.35896949, 0.6559765 ],
       [0.24501778, 0.35458782, 0.65241923],
       [0.24526992, 0.35022239, 0.64873121],
       [0.24551877, 0.34587559, 0.64490987],
       [0.24575701, 0.34154984, 0.64095315],
       [0.24597735, 0.33724755, 0.63685958],
       [0.24617258, 0.33297109, 0.6326283 ],
       [0.24633566, 0.32872279, 0.62825909],
       [0.24645977, 0.32450489, 0.6237524 ],
       [0.2465384 , 0.32031951, 0.61910933],
       [0.24656536, 0.31616862, 0.61433167],
       [0.2465349 , 0.31205404, 0.60942183],
       [0.2464417 , 0.30797739, 0.60438286],
       [0.24628095, 0.30394008, 0.59921837],
       [0.24604835, 0.29994331, 0.59393251],
       [0.24574017, 0.29598806, 0.58852986],
       [0.24535323, 0.29207505, 0.5830154 ],
       [0.2448849 , 0.28820479, 0.57739441],
       [0.24433314, 0.28437755, 0.5716724 ],
       [0.24369643, 0.2805934 , 0.56585499],
       [0.24297377, 0.2768522 , 0.55994789],
       [0.24216465, 0.27315361, 0.55395673],
       [0.24126904, 0.26949715, 0.54788707],
       [0.24028265, 0.26588249, 0.54175028],
       [0.23920692, 0.26230862, 0.53555174],
       [0.23804591, 0.25877435, 0.529293  ],
       [0.23680103, 0.25527869, 0.52297876],
       [0.23547394, 0.25182056, 0.51661336],
       [0.23406342, 0.2483988 , 0.5102062 ],
       [0.23256539, 0.24501182, 0.50377332],
       [0.23099051, 0.24165846, 0.49730374],
       [0.22934111, 0.23833757, 0.49080012],
       [0.22761504, 0.23504748, 0.48427492],
       [0.22580933, 0.23178597, 0.47774457],
       [0.22393594, 0.22855285, 0.47119005],
       [0.22199703, 0.22534707, 0.46461293],
       [0.21998016, 0.22216408, 0.45805652],
       [0.21790275, 0.21900569, 0.45148217],
       [0.21576549, 0.2158706 , 0.44489464],
       [0.21355801, 0.2127534 , 0.43833741],
       [0.21129685, 0.20965752, 0.43176498],
       [0.208977  , 0.20657923, 0.42520322],
       [0.20659892, 0.20351633, 0.41866115],
       [0.20417332, 0.2004721 , 0.41210346],
       [0.20168703, 0.19743702, 0.40559795],
       [0.19915672, 0.19441854, 0.39907853],
       [0.19657462, 0.19140947, 0.39259226],
       [0.19394722, 0.188412  , 0.38611587],
       [0.19127464, 0.18542401, 0.37965796],
       [0.18855724, 0.18244322, 0.37322757],
       [0.18579938, 0.17947155, 0.36680604],
       [0.18299849, 0.17650348, 0.36042297],
       [0.18016036, 0.17354351, 0.35404476],
       [0.1772819 , 0.17058464, 0.34770888],
       [0.17436826, 0.16763191, 0.34137988],
       [0.17141745, 0.16467906, 0.33508977],
       [0.1684328 , 0.16172931, 0.32881489],
       [0.16541424, 0.15877962, 0.32256801],
       [0.16236275, 0.15582876, 0.31635122],
       [0.15928037, 0.15287968, 0.31014411],
       [0.15616589, 0.14992378, 0.30398848],
       [0.15302294, 0.14697296, 0.29781705],
       [0.14984897, 0.14400828, 0.29172467],
       [0.14664749, 0.14104684, 0.28561838],
       [0.14341772, 0.13807648, 0.27955643],
       [0.14016072, 0.13510078, 0.27351646],
       [0.13687687, 0.13212284, 0.26747923],
       [0.13356695, 0.12912934, 0.26150605],
       [0.13023079, 0.12613483, 0.25552177],
       [0.12686948, 0.1231272 , 0.24958092],
       [0.12348311, 0.12010958, 0.24366409],
       [0.12007068, 0.11708837, 0.23773723],
       [0.11663557, 0.11404484, 0.23188363],
       [0.11317482, 0.11099513, 0.22602364],
       [0.1096889 , 0.10793517, 0.22017219],
       [0.10618075, 0.1048534 , 0.21437716],
       [0.10264643, 0.10176328, 0.2085729 ],
       [0.0990876 , 0.09865736, 0.20278851],
       [0.09550606, 0.09552937, 0.19704663],
       [0.09189749, 0.09238992, 0.19129533],
       [0.08826408, 0.08923087, 0.18556525],
       [0.08460723, 0.0860472 , 0.17987258],
       [0.08092191, 0.08284857, 0.17416973],
       [0.07720982, 0.0796282 , 0.16848004],
       [0.07347361, 0.07637816, 0.16283083],
       [0.0697067 , 0.0731091 , 0.15717011],
       [0.06590891, 0.0698176 , 0.15150562],
       [0.06208647, 0.06648866, 0.14589217],
       [0.05823   , 0.06313586, 0.14026517],
       [0.0543378 , 0.05975738, 0.13462392],
       [0.05041622, 0.05633789, 0.12902168],
       [0.04645778, 0.0528849 , 0.1234176 ],
       [0.0424579 , 0.04939964, 0.11779608],
       [0.0384136 , 0.04587404, 0.11217763],
       [0.03450984, 0.04229956, 0.10658282],
       [0.03080428, 0.03867708, 0.10096627],
       [0.02729776, 0.03514909, 0.09532647],
       [0.02399819, 0.03176264, 0.0897075 ],
       [0.02090133, 0.02852652, 0.08407772],
       [0.0180055 , 0.02544552, 0.07841879],
       [0.01531167, 0.02252059, 0.07272874]]

    white_ice_map = LinearSegmentedColormap.from_list('white_ice', cmdata)
    return white_ice_map

def white_dense():
    # cdf = pd.DataFrame(cmap(np.linspace(0,1,256)))
    # cdf.to_csv('colors.csv',index=False,header=False)
    cmdata = [[1 , 1, 1, 1],
[0.9022021640633742,0.9441797977915001,0.9438027309131503,1.0],
[0.8954445384876382,0.9409578903665055,0.9410648759794843,1.0],
[0.8886855788874896,0.937740382063468,0.9384098650760987,1.0],
[0.8819275100483496,0.9345262546442011,0.935837281404758,1.0],
[0.8751724774625698,0.931314552161322,0.9333465385891809,1.0],
[0.8684225465562396,0.9281043788756372,0.9309368983040341,1.0],
[0.8616797015669134,0.9248948973050556,0.9286074874996731,1.0],
[0.8549458444869071,0.921685326259987,0.9263573151615366,1.0],
[0.8482227943572994,0.9184749387758169,0.9241852885114791,1.0],
[0.8415122870990739,0.9152630598938195,0.9220902285443439,1.0],
[0.8348159759916424,0.9120490642716417,0.9200708847923535,1.0],
[0.8281354328507814,0.9088323736256948,0.918125949218801,1.0],
[0.8214721499137223,0.9056124540224106,0.9162540691578,1.0],
[0.8148275424063925,0.9023888130447766,0.9144538592359539,1.0],
[0.8082029517444305,0.8991609968660188,0.912723912232588,1.0],
[0.8015996493038703,0.8959285872646755,0.9110628088561002,1.0],
[0.7950188742001107,0.8926911903474541,0.9094690972400389,1.0],
[0.7884617893745216,0.8894484461080552,0.9079413399350438,1.0],
[0.781929470922247,0.8862000288332801,0.9064781367807945,1.0],
[0.7754229463217647,0.8829456358148736,0.9050781031537078,1.0],
[0.7689431984424602,0.879684984924093,0.9037398757679528,1.0],
[0.7624911712284808,0.8764178119047121,0.9024621162700895,1.0],
[0.756067775420235,0.8731438677954836,0.901243514034475,1.0],
[0.749673894265701,0.8698629164913183,0.9000827882281618,1.0],
[0.7433103891809361,0.8665747324492172,0.8989786892140232,1.0],
[0.7369781053261772,0.8632790985421133,0.897929999359423,1.0],
[0.7306778770702587,0.8599758040613866,0.8969355333152028,1.0],
[0.7244105333220341,0.856664642866711,0.8959941378263615,1.0],
[0.7181769027125698,0.8533454116802826,0.8951046911317568,1.0],
[0.7119778186163851,0.850017908521137,0.8942661020057924,1.0],
[0.705814228526864,0.8466819129479339,0.8934771781429077,1.0],
[0.6996868754737947,0.843337242563576,0.8927370216147951,1.0],
[0.6935966262281187,0.8399836925869731,0.8920446369253885,1.0],
[0.6875443783072674,0.8366210531289836,0.8913990435590288,1.0],
[0.6815310569542375,0.8332491095447561,0.8907992825434408,1.0],
[0.6755576190661271,0.8298676415986731,0.8902444142113377,1.0],
[0.6696250569512002,0.8264764227239829,0.889733515850932,1.0],
[0.6637344019158556,0.8230752193711802,0.8892656792634948,1.0],
[0.6578867276821276,0.8196637904396378,0.8888400082431818,1.0],
[0.652083153635169,0.8162418867874922,0.888455615991751,1.0],
[0.6463248478988037,0.8128092508153052,0.8881116224784992,1.0],
[0.6406130302355731,0.8093656161195516,0.8878071517537653,1.0],
[0.634948974765717,0.8059107072125803,0.8875413292226666,1.0],
[0.6293340124974294,0.8024442393062223,0.8873132788842952,1.0],
[0.6237695336583948,0.7989659181568091,0.887122120540472,1.0],
[0.6182569898160699,0.7954754399699002,0.8869669669772311,1.0],
[0.6127978529026838,0.7919724939438573,0.8868470036964815,1.0],
[0.6073937143892588,0.7884567552174954,0.8867613059311603,1.0],
[0.602046236220424,0.7849278896215749,0.886708922592684,1.0],
[0.5967571335484213,0.7813855546115022,0.8866889106706612,1.0],
[0.5915281898823177,0.7778293984625484,0.886700305501537,1.0],
[0.5863612575196054,0.7742590605302299,0.8867421178165009,1.0],
[0.5812582575983973,0.7706741715732086,0.8868133307743576,1.0],
[0.57622117973401,0.7670743541409147,0.8869128969829141,1.0],
[0.5712520812007982,0.7634592230284989,0.8870397355132994,1.0],
[0.5663530856178774,0.75982838580195,0.8871927289126763,1.0],
[0.561526381094974,0.7561814433965411,0.8873707202219829,1.0],
[0.5567742177930776,0.7525179907919229,0.8875725100066919,1.0],
[0.5520989048532665,0.7488376177673575,0.8877968534100384,1.0],
[0.5475028066465009,0.7451399097406839,0.8880424572398105,1.0],
[0.5429883038774732,0.7414244411965056,0.8883081102246548,1.0],
[0.5385578037806127,0.7376907727350942,0.8885926739767255,1.0],
[0.5342139074650678,0.7339384990781334,0.8888943435724942,1.0],
[0.5299591516928193,0.730167197615884,0.8892116013883854,1.0],
[0.5257960979209626,0.7263764470411207,0.8895428673545512,1.0],
[0.5217273234975481,0.7225658287807644,0.8898864965652431,1.0],
[0.5177554119038458,0.7187349285333119,0.8902407770595433,1.0],
[0.5138829420298184,0.7148833379130645,0.8906039278005389,1.0],
[0.510112476480756,0.7110106562012711,0.8909740968830554,1.0],
[0.5064465489258501,0.7071164922032952,0.8913493600019623,1.0],
[0.5028876505140545,0.7032004662097455,0.8917277192147863,1.0],
[0.49943821525823867,0.6992621674331674,0.8921074350832908,1.0],
[0.49610063472159843,0.6953012224940205,0.8924864661543193,1.0],
[0.4928771955532801,0.69131735422067,0.8928621485317783,1.0],
[0.4897700668589556,0.6873102514205023,0.8932321753875414,1.0],
[0.48678129437860457,0.6832796259123631,0.8935941653626673,1.0],
[0.4839127827295982,0.6792252150037094,0.8939456639464464,1.0],
[0.48116627742448465,0.675146784009974,0.8942841454160184,1.0],
[0.47854334684713423,0.6710441287994943,0.8946070153685666,1.0],
[0.47604536438908757,0.6669170783452496,0.8949116138746747,1.0],
[0.47367364282946584,0.6627653768113492,0.895195785971628,1.0],
[0.4714290262412599,0.6585890168305761,0.8954562723025643,1.0],
[0.4693121574935592,0.654387975766499,0.8956900818434682,1.0],
[0.4673234528945733,0.6501622438680934,0.8958943277799077,1.0],
[0.4654630534005659,0.6459118568983728,0.8960660842179662,1.0],
[0.46373081267883526,0.6416368981971656,0.8962023935259622,1.0],
[0.46212628700852026,0.6373375005637915,0.8963002743257549,1.0],
[0.4606487857322991,0.63301381556304,0.896356841443646,1.0],
[0.4592973162762511,0.6286660463835676,0.8963691878774374,1.0],
[0.45807033392873825,0.6242945769689846,0.8963339924500021,1.0],
[0.4569661322748619,0.6198997574323486,0.8962482674873512,1.0],
[0.4559826907629682,0.615481993049396,0.8961090514006179,1.0],
[0.45511767935042186,0.6110417444047529,0.8959134196000665,1.0],
[0.4543684658154032,0.6065795271800728,0.8956584955889488,1.0],
[0.45373211030722954,0.6020959187639063,0.8953414439077736,1.0],
[0.4532053056792195,0.5975915899334859,0.8949594061016021,1.0],
[0.45278458048498743,0.5930672153918073,0.8945097368602322,1.0],
[0.4524662055084719,0.5885235252040262,0.8939898705469653,1.0],
[0.4522462243009037,0.5839612972830612,0.8933973456284665,1.0],
[0.4521204742039108,0.5793813548677186,0.8927298144348157,1.0],
[0.4520846088132207,0.5747845636677534,0.89198505230055,1.0],
[0.452134064027873,0.5701718535137081,0.8911609204213973,1.0],
[0.45226365565476695,0.5655443967339887,0.8902550664458119,1.0],
[0.45246908216723686,0.5609029696513449,0.8892660746141661,1.0],
[0.4527455061379137,0.5562485728577384,0.8881923021329056,1.0],
[0.45308803206591247,0.5515822303923639,0.88703226279663,1.0],
[0.4534917310405355,0.546904985561765,0.8857846304506886,1.0],
[0.45395166471791165,0.5422178966900856,0.8844482413776628,1.0],
[0.454462908369541,0.5375220328430568,0.883022095602165,1.0],
[0.45502044436875183,0.5328185230036192,0.8815052933071011,1.0],
[0.45561856110926985,0.5281088097066268,0.8798967638851668,1.0],
[0.45625342042665423,0.5233935859930701,0.8781964868264528,1.0],
[0.45692037090845966,0.5186739169401045,0.8764041195297783,1.0],
[0.45761487064778333,0.513950857677444,0.8745194730542405,1.0],
[0.4583325010790183,0.5092254498718805,0.8725425074817807,1.0],
[0.4590689790493137,0.504498718441123,0.8704733265502939,1.0],
[0.4598201671313326,0.49977166851758464,0.8683121716406742,1.0],
[0.4605820822069772,0.49504528267844344,0.866059415204812,1.0],
[0.461350902372687,0.4903205184540396,0.8637155537234259,1.0],
[0.46212297223505533,0.4855983061224402,0.8612812002829223,1.0],
[0.4628948066804312,0.48087954679397343,0.8587570768592888,1.0],
[0.46366216178047465,0.4761655004408037,0.8561438461496413,1.0],
[0.464422848334832,0.471456609607848,0.8534426324219575,1.0],
[0.4651739386269665,0.466753663570378,0.8506544391441876,1.0],
[0.46591264858848286,0.4620574294334567,0.8477803402103137,1.0],
[0.4666363628095653,0.45736864021084417,0.8448214774526085,1.0],
[0.46734263133213166,0.4526879945693309,0.8417790529251528,1.0],
[0.4680291657078233,0.44801615679558254,0.8386543213914651,1.0],
[0.468693834428531,0.4433537569692766,0.8354485830591045,1.0],
[0.4693346578320167,0.4387013913255956,0.83216317659744,1.0],
[0.4699498025791529,0.4340596227897761,0.8287994724682681,1.0],
[0.47053757579268535,0.4294289816663296,0.8253588665927665,1.0],
[0.4710964189403291,0.4248099664657251,0.8218427743723752,1.0],
[0.4716249015377835,0.42020304485173476,0.8182526250758133,1.0],
[0.47212171473984715,0.41560865469322217,0.8145898565994109,1.0],
[0.4725856648806062,0.4110272052049121,0.8108559106034937,1.0],
[0.47301566701650727,0.4064590781625462,0.8070522280235293,1.0],
[0.4734107385193607,0.40190462917879616,0.8031802449512829,1.0],
[0.47376999275981607,0.39736418902732973,0.7992413888782302,1.0],
[0.47409263291575826,0.39283806500351043,0.7952370752909331,1.0],
[0.4743779459344564,0.38832654231127745,0.7911687046060347,1.0],
[0.4746252966720312,0.38382988546686225,0.7870376594308572,1.0],
[0.4748341222291213,0.3793483397110574,0.7828453021343189,1.0],
[0.4750039264972927,0.37488213242278556,0.7785929727119724,1.0],
[0.475134274926932,0.37043147452773884,0.7742819869283568,1.0],
[0.4752247895239137,0.36599656189678464,0.7699136347195339,1.0],
[0.4752751440794078,0.3615775767297375,0.7654891788385849,1.0],
[0.47528465773460166,0.3571748688576533,0.7610100397104845,1.0],
[0.4752531583445196,0.35278856956956184,0.7564774234071201,1.0],
[0.4751807975405667,0.3484186721552498,0.7518923528283415,1.0],
[0.47506742066531155,0.34406531404041674,0.7472559685686854,1.0],
[0.47491290421939053,0.33972862566776674,0.7425693796885003,1.0],
[0.4747171524182527,0.3354087317676911,0.7378336635067139,1.0],
[0.4744800939620511,0.3311057526021228,0.733049865530307,1.0],
[0.47420167901053806,0.32681980518219417,0.7282189995072744,1.0],
[0.47388187635431483,0.3225510044607446,0.7233420475907059,1.0],
[0.47352067077346277,0.3182994645010272,0.7184199606025196,1.0],
[0.4731180605743212,0.31406529962333257,0.7134536583862346,1.0],
[0.472674055295149,0.3098486255314443,0.7084440302390294,1.0],
[0.47218867357134314,0.30564956042113184,0.703391935414181,1.0],
[0.47166137272970204,0.3014684828248803,0.6982986813805611,1.0],
[0.4710926963645385,0.29730528702668124,0.6931646771793223,1.0],
[0.470482813628966,0.29316004418388464,0.6879905802658635,1.0],
[0.46983176393818465,0.28903289184955194,0.6827771264871377,1.0],
[0.4691395866583317,0.28492397574667777,0.677525024847434,1.0],
[0.46840631973819363,0.2808334508365701,0.6722349583426855,1.0],
[0.4676319984415455,0.2767614823885942,0.6669075848245407,1.0],
[0.4668166541724163,0.27270824705461943,0.6615435378909207,1.0],
[0.46596031338591853,0.26867393395157463,0.6561434278004181,1.0],
[0.4650628643816827,0.2646588038897797,0.6507079958832593,1.0],
[0.4641242580497153,0.2606631006122758,0.6452378997676681,1.0],
[0.4631447877428096,0.25668693057016545,0.6397333527350717,1.0],
[0.46212444946943126,0.25273054493608815,0.6341948722101477,1.0],
[0.46106322933938265,0.2487942133948352,0.6286229575018943,1.0],
[0.45996110285585984,0.24487822535483766,0.6230180909006084,1.0],
[0.4588180342503813,0.24098289119188765,0.6173807387973655,1.0],
[0.45763397585548965,0.23710854352850097,0.6117113528277586,1.0],
[0.45640886751035886,0.23325553855222453,0.6060103710422048,1.0],
[0.4551426114844221,0.22942426724544046,0.600278257550051,1.0],
[0.4538352109035017,0.22561510094159637,0.5945152849064859,1.0],
[0.4524865766439375,0.22182847164920177,0.5887218272221995,1.0],
[0.4510965881023942,0.21806484773466195,0.5828982757017971,1.0],
[0.4496651090864017,0.2143247287775244,0.5770450152504881,1.0],
[0.44819198734394855,0.21060864716511674,0.571162425977021,1.0],
[0.4466770541000938,0.20691716973621027,0.5652508847626542,1.0],
[0.4451201235978615,0.20325089947246439,0.5593107669047255,1.0],
[0.4435210649059648,0.19961045356829826,0.5533422989220041,1.0],
[0.44187975508877425,0.19599648480063536,0.5473456323671032,1.0],
[0.44019583038684174,0.1924097603751146,0.5413213827196031,1.0],
[0.4384690252866834,0.18885105108443032,0.535269939274394,1.0],
[0.43669905473379284,0.1853211730457039,0.529191700905823,1.0],
[0.43488561370620266,0.18182098950074954,0.523087078501802,1.0],
[0.43302837679171846,0.17835141259016973,0.5169564975681201,1.0],
[0.4311269977716754,0.17491340507780434,0.5108004010215621,1.0],
[0.4291811092153304,0.17150798199683612,0.5046192521919093,1.0],
[0.42719043446599236,0.16813619679249853,0.49841321065711747,1.0],
[0.42515465226735816,0.16479917559522433,0.49218247403091814,1.0],
[0.42307312862861274,0.16149813597846996,0.48592813807538265,1.0],
[0.42094540041362066,0.15823432004427912,0.4796507952549767,1.0],
[0.4187709818950797,0.15500902723217586,0.47335107912582347,1.0],
[0.41654936469111414,0.15182361435195332,0.4670296690947608,1.0],
[0.41428001780800694,0.14867949510324718,0.46068729553509385,1.0],
[0.41196238781455147,0.14557813897896077,0.45432474527733707,1.0],
[0.40959589917786626,0.14252106943690282,0.4479428674896457,1.0],
[0.4071799547953331,0.13950986121109918,0.44154257995746904,1.0],
[0.40471393676249223,0.13654613662168955,0.43512487576483,1.0],
[0.40219729360192846,0.1336316105494001,0.4286903779176353,1.0],
[0.39962936308492736,0.13076801503504104,0.42224018363382726,1.0],
[0.3970093606178293,0.1279570274853844,0.41577611200457687,1.0],
[0.3943365976380543,0.125200394269635,0.40929955131196355,1.0],
[0.39161037490335765,0.12249986881235914,0.40281200413413176,1.0],
[0.38882998588400647,0.11985719841100997,0.39631509549997207,1.0],
[0.38599472080598896,0.11727410895472495,0.38981058099735255,1.0],
[0.3831038714159067,0.11475228744179017,0.38330035466997037,1.0],
[0.3801567365353384,0.11229336223745998,0.37678645649994913,1.0],
[0.3771526284658991,0.109898881072994,0.3702710792321893,1.0],
[0.37409088029500187,0.10757028686219475,0.36375657425343866,1.0],
[0.37097085413549286,0.10530889150402406,0.35724545619579434,1.0],
[0.3677919503089708,0.10311584794863485,0.3507404058933542,1.0],
[0.36455361745195075,0.10099212092759274,0.34424427128517043,1.0],
[0.3612553634855837,0.09893845688361261,0.33776006583143864,1.0],
[0.3578967673433453,0.0969553537752958,0.3312909639976104,1.0],
[0.35447749129751766,0.09504303157033087,0.324840293367848,1.0],
[0.3509972936657821,0.0932014043666256,0.31841152298020636,1.0],
[0.3474560416161082,0.09143005518316707,0.3120082475359999,1.0],
[0.34385372372478934,0.08972821452832538,0.30563416722884074,1.0],
[0.3401904735486729,0.08809468358171338,0.29929318157551454,1.0],
[0.33646657247147094,0.08652789261304067,0.2929892038941975,1.0],
[0.33268245026470633,0.08502592614588003,0.28672608390538185,1.0],
[0.3288387024976534,0.0835864705108664,0.28050768472776955,1.0],
[0.3249360934730276,0.08220683878873383,0.27433781519411793,1.0],
[0.32097555841016556,0.0808839910074109,0.26822018715503526,1.0],
[0.31695820222362264,0.07961456284429594,0.26215837131986064,1.0],
[0.31288529460128056,0.07839490235071095,0.2561557528820557,1.0],
[0.3087582612248005,0.07722111381434071,0.25021548831259177,1.0],
[0.3045786711401688,0.07608910749927883,0.24434046477148758,1.0],
[0.30034822046705684,0.07499465368295166,0.23853326357172805,1.0],
[0.29606871281984143,0.07393343917254722,0.23279612902452418,1.0],
[0.29174203698566603,0.07290112435224017,0.2271309438018545,1.0],
[0.28737014255119664,0.07189339879884682,0.22153921168164575,1.0],
[0.28295501427674163,0.07090603360902006,0.21602204821131593,1.0],
[0.2784999277173712,0.06993210653137302,0.21058327359946222,1.0],
[0.2740068130913843,0.06896804327168699,0.20522255483273655,1.0],
[0.2694770905593399,0.06801140076763185,0.19993831225037145,1.0],
[0.2649127155611373,0.06705861363954488,0.1947301490341073,1.0],
[0.260315555546597,0.06610638194498253,0.18959732755599618,1.0],
[0.25568737133504715,0.06515168315524647,0.18453880032644063,1.0],
[0.2510311382384793,0.06418928564209234,0.17955550075065288,1.0],
[0.24635152648506936,0.06321115386076698,0.17465055684091405,1.0],
[0.2416465428487372,0.06222162722501617,0.16981600423165008,1.0],
[0.23691745126961133,0.061218792196168555,0.1650498029374761,1.0],
[0.23216535560711132,0.06020099184329732,0.16034977404957257,1.0],
[0.22739756443587622,0.05915624173942468,0.15572184565633526,1.0],
[0.2226121431237117,0.058088424655696934,0.151159081535817,1.0],
[0.2178074381899826,0.0570004979074318,0.146655818703497,1.0],
[0.21298394220008482,0.05589168645699472,0.14220950682407338,1.0],
[0,0,0,1]
]

    white_ice_map = LinearSegmentedColormap.from_list('white_dense', cmdata)
    return white_ice_map