from transformer_model import utils_runner_universal
from transformer_model.utils_problem_classification_processor import *
from transformer_model.utils import *
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from tqdm import tqdm
from config import *
import argparse





parser = argparse.ArgumentParser(description="A script that creates a list of functions based on the input arguments")
parser.add_argument("--dimension", type=int, help="The problem dimension")

parser.add_argument("--fold", type=int, help="The training fold")

parser.add_argument("--sample_count_dimension_factor", type=int, help="The number of samples to generate will be set to sample_count_dimension_factor*dimension")
parser.add_argument("--instances_to_use", type=int, help="The number of instances to use from each problem class")

args = parser.parse_args()
dimension = args.dimension
sample_count_dimension_factor=args.sample_count_dimension_factor
fold=args.fold
instances_to_use=args.instances_to_use

x_columns=[f'x_{i}' for i in range (0,dimension)]
y_columns = ['y']



def read_sample_df(dimension, sample_count_dimension_factor):
    '''file_to_read = open(f"data/lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.p", "rb")
    loaded_dictionary = pickle.load(file_to_read)
    all_sample_df=pd.DataFrame.from_dict(loaded_dictionary,orient='index')
    all_sample_df.columns=x_columns+y_columns
    all_sample_df.index=pd.MultiIndex.from_tuples(all_sample_df.index, names=['problem_id', 'instance_id','dimension','sample_id'])'''
    all_sample_df=pd.read_csv(f'{data_dir}/transformer_training_samples/lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.csv', index_col=['f'])
    return all_sample_df

def scale_y(all_sample_df):
    print("Scaling")
    for f in tqdm(all_sample_df.reset_index()['f'].drop_duplicates()):


        min_max_scaler = MinMaxScaler()
        all_sample_df.loc[f,'y'] = [x[0] for x in list(min_max_scaler.fit_transform(all_sample_df.loc[f,'y'].values.reshape(-1, 1)))]



    return all_sample_df

scaled_file=f"{data_dir}/transformer_training_samples/scaled_lhs_samples_dimension_{dimension}_{sample_count_dimension_factor}_samples.csv"
print(scaled_file)
if not os.path.isfile(scaled_file):

    all_sample_df=read_sample_df(dimension, sample_count_dimension_factor)
    new_sample_df=scale_y(all_sample_df)
    new_sample_df.to_csv(scaled_file)
else:
    print('Reading from file')
    new_sample_df=pd.read_csv(scaled_file, index_col=[0])
    
new_sample_df=new_sample_df.loc[[f'{p}_{i}' for p in range (1,25) for i in range(1,instances_to_use+1)],:]
splitter=SplitterBBOBInstance(include_tuning=False)
data_processor=ProblemClassificationProcessor(verbose=False,fold=fold,split_ids_dir=f'problem_classification_{instances_to_use}_instances', id_names=['f'], splitter=splitter)
print(new_sample_df)
for d_model in [30]:
    for n_heads in [1]:
        for n_layers in [1]:
                sample_df=new_sample_df.copy()
                utils_runner_universal.UniversalRunner(data_processor, global_result_dir='results_trash',extra_info=f'dim_{dimension}_instances_{instances_to_use}_samples_{sample_count_dimension_factor}',use_positional_encoding=False, verbose=True, plot_training=False, d_model=d_model, d_k=None, d_v=None, n_heads=n_heads, n_layers=n_layers, n_epochs=200,  lr_max=0.001, id_names=['f']).run(sample_df, regenerate=True)






