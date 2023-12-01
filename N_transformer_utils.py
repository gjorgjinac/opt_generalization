from transformer_model import utils_runner_universal
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sklearn
import seaborn as sns
import colorcet as cc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformer_model.tsai_custom import *
from transformer_model.model_stats import *
from tsai.all import *
from config import *
computer_setup()
from tqdm import tqdm

import argparse
from N_sampling_utils import *
from config import *

def get_embeddings_from_trained_model(model, dls,cast_y_to_int=True):
    all_embeddings=None
    all_labels=[]

    for batch in dls:
        batch_embeddings=model.cuda().get_embeddings(batch[0].cuda())
        batch_embeddings=batch_embeddings.detach().cpu().numpy()
        all_embeddings=batch_embeddings if all_embeddings is None else np.append(all_embeddings,batch_embeddings, axis=0)
        all_labels+=list(batch[1])
    if cast_y_to_int:
            all_labels=[int(i) for i in all_labels]
    return all_embeddings, all_labels
    
    
def get_x_y(sample_df, shuffle=False):
    xys=[]
    for function in sample_df.index.drop_duplicates().values:
        function_instances=sample_df.loc[function]
        xys+=[(function_instances.values,function)]
    if shuffle:
        random.shuffle(xys)
    x=np.array([np.array(xy[0]) for xy in xys])
    y=np.array([xy[1] for xy in xys])
    return x,y

def scale_y(sample_df,verbose=True):
    new_sample_df=pd.DataFrame()
    if verbose:
        for function in tqdm(sample_df.index.drop_duplicates().values):
            try:
                instance_df=sample_df.loc[function].copy()
                min_max_scaler = MinMaxScaler()

                y_scaled = min_max_scaler.fit_transform(instance_df['y'].values.reshape(-1, 1))
                instance_df.loc[:,'y']=y_scaled
                instance_df['f']=function
                new_sample_df=pd.concat([new_sample_df,instance_df])
            except Exception:
                print('failed minmax', function)
                continue
    else:
        for function in sample_df.index.drop_duplicates().values:
            try:
                instance_df=sample_df.loc[function].copy()
                min_max_scaler = MinMaxScaler()

                y_scaled = min_max_scaler.fit_transform(instance_df['y'].values.reshape(-1, 1))
                instance_df.loc[:,'y']=y_scaled
                instance_df['f']=function
                new_sample_df=pd.concat([new_sample_df,instance_df])
            except Exception:
                print('failed minmax', function)
                continue
            
    new_sample_df=new_sample_df.set_index(['f'])
    return new_sample_df

def load_model_problem_classification(fold,dimension,sample_count_dimension_factor):
    model_path=f'results_downstream/problem_classification_stats/dim_{dimension}_instances_999_samples_{sample_count_dimension_factor}_fold_{fold}_n_heads_1_n_layers_1_d_model_30_d_k_None_d_v_None_aggregations_all/trained_model_dict.pt'
    model =OptTransStats(dimension+1, 24, sample_count_dimension_factor, n_heads=1, n_layers=1, d_model=30, d_k=None, d_v=None, use_positional_encoding=False, iteration_count=None, aggregations=None)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_model_performance_prediction(fold,dimension,sample_count_dimension_factor,algorithm_portfolio,train_benchmark):
    model_path=f'results_downstream/performance_prediction_stats/dim_{dimension}_instances_999_samples_{sample_count_dimension_factor}_fold_{fold}_n_heads_1_n_layers_1_d_model_30_d_k_None_d_v_None_aggregations_all/{algorithm_portfolio}_{train_benchmark}/trained_model_dict.pt'
    model =OptTransStats(dimension+1, len(algorithm_portfolio.split('-')), sample_count_dimension_factor, n_heads=1, n_layers=1, d_model=30, d_k=None, d_v=None, use_positional_encoding=False, iteration_count=None, aggregations=None)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model