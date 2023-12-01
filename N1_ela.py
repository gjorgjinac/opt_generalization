import pandas as pd
from pflacco.classical_ela_features import *
import sys
from config import *
import argparse
from N_ranking_utils import calculate_ela
from tqdm import tqdm
parser = argparse.ArgumentParser(description="A script that runs the algorithms on all benchmark functions")


parser.add_argument("--dimension", type=int, help="The problem dimension")
parser.add_argument("--benchmark", type=str, help="Benchmark name")
parser.add_argument("--sample_count_dimension_factor", type=int, help="The number of samples to generate will be set to sample_count_dimension_factor*dimension")

args = parser.parse_args()
benchmark=args.benchmark
dimension=args.dimension
df = pd.read_csv(f'{data_dir}/samples/{args.sample_count_dimension_factor}d_samples/{benchmark}_{dimension}d.csv', index_col=0)
if 'f' in df.columns:
    df=df.set_index('f')
else:
    df.index.name='f'

df=df.replace([np.inf,-np.inf],np.nan)
x=df.isna().sum(axis=1).reset_index().groupby('f').sum()
x.columns=['count']
functions_to_keep=list(x.query('count==0').index)
df=df.query('f in @functions_to_keep')


ela_functions_x_y_parameters = [calculate_dispersion, calculate_ela_distribution, calculate_ela_level, calculate_ela_meta, calculate_information_content, calculate_nbc, calculate_pca]
ela_functions_x_y_lower_bound_upper_bound_parameters = [calculate_cm_angle, calculate_cm_conv, calculate_cm_grad, calculate_limo] 
ela_functions_x_y_f_parameters = [calculate_ela_conv]
                                  
ela_functions_x_y_lower_bound_upper_bound_f_dim_parameters = [calculate_ela_curvate, calculate_ela_local]


all_ela_features=[]

for f in tqdm(list(df.index.drop_duplicates().values)):
    f_samples=df.loc[f]
    X = f_samples[[f'x_{i}' for i in range(0,dimension)]]
    print(X.shape)
    Y = f_samples['y']
    f_ela_features=calculate_ela(X,Y)
    f_ela_features['f']=f
    all_ela_features+=[f_ela_features]


all_ela_features=pd.DataFrame(all_ela_features)
all_ela_features=all_ela_features.set_index('f')
all_ela_features.to_csv(f'{data_dir}/pela_features/{args.sample_count_dimension_factor}d_samples/{benchmark}_{dimension}d.csv')
                                                              