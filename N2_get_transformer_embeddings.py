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
from N_transformer_utils import *

parser = argparse.ArgumentParser(description="A script that creates a list of functions based on the input arguments")
parser.add_argument("--dimension", type=int, help="The problem dimension")
parser.add_argument("--benchmark", type=str, help="The benchmark for which to generate embeddings")
parser.add_argument("--sample_count_dimension_factor", type=int, help="The number of samples to generate will be set to sample_count_dimension_factor*dimension")
parser.add_argument("--fold", type=int, help="The training fold")

parser.add_argument("--problem_classification", type=int, help="1 if the training task is problem classification", default=1, required=False)
parser.add_argument("--algorithms", type=str, help="Algorithm names separated by -. Required if problem_classification parameters is set to 0", required=False)
parser.add_argument("--train_benchmark", type=str, help="Benchmark on which the performance prediction model was trained. Required if problem_classification parameters is set to 0", required=False)


args = parser.parse_args()
dimension=args.dimension
benchmark=args.benchmark
fold=args.fold
sample_count_dimension_factor=args.sample_count_dimension_factor
instances_to_use=999
x_columns=[f'x_{i}' for i in range (0,dimension)]
y_columns = ['y']


scaled_sample_file=f'{data_dir}/samples/{sample_count_dimension_factor}d_samples/{benchmark}_{dimension}d_scaled.csv'
if not os.path.isfile(scaled_sample_file):
    print('rescaling samples')
    sample_df=pd.read_csv(f'{data_dir}/samples/{sample_count_dimension_factor}d_samples/{benchmark}_{dimension}d.csv',index_col=0)
    if 'f' in sample_df.columns:
        sample_df=sample_df.set_index('f')
    print(sample_df.shape)
    sample_df_scaled=scale_y(sample_df)
    sample_df_scaled.to_csv(scaled_sample_file)
else:
    print('reading scaled samples')
    sample_df_scaled=pd.read_csv(scaled_sample_file,index_col=0)
x,y=get_x_y(sample_df_scaled)
function_to_id={f:i for i,f in enumerate(y)}
id_to_function={i:f for f,i in function_to_id.items()}
y_ids= [function_to_id[f] for f in y]
dset = TSDatasets(np.swapaxes(x,1,2),y_ids)
dls = TSDataLoaders.from_dsets(dset, bs=50)

if args.problem_classification==1:
    model=load_model_problem_classification(fold,dimension,sample_count_dimension_factor)
else:
    model=load_model_performance_prediction(fold,dimension,sample_count_dimension_factor,args.algorithms,args.train_benchmark)
e=get_embeddings_from_trained_model(model, dls[0])
embeddings=pd.DataFrame(e[0], index=[id_to_function[i] for i in e[1]])
save_dir = 'transformer_features' if args.problem_classification==1 else 'transformer_performance_prediction_features'
os.makedirs(f'{data_dir}/{save_dir}/{sample_count_dimension_factor}d_samples', exist_ok=True)
embeddings.to_csv(f'{data_dir}/{save_dir}/{sample_count_dimension_factor}d_samples/{benchmark}_{dimension}d_fold_{fold}.csv')