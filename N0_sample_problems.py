import argparse
from N_sampling_utils import *
from config import *
parser = argparse.ArgumentParser(description="A script that creates a list of functions based on the input arguments")


parser.add_argument("--dimension", type=int, help="The problem dimension")
parser.add_argument("--random_problem_count", type=int, help="The number of random problems to generate")
parser.add_argument("--m4_problem_count", type=int, help="The number of M4 problems to generate")
parser.add_argument("--sample_count_dimension_factor", type=int, help="The number of samples to generate will be set to sample_count_dimension_factor*dimension")

args = parser.parse_args()
sample_range=None
os.makedirs(f'{data_dir}/samples/{args.sample_count_dimension_factor}d_samples', exist_ok=True)
generate_and_sample_random_functions(dimension=args.dimension,sample_count_dimension_factor=args.sample_count_dimension_factor,n_functions=args.random_problem_count, sample_range=sample_range)
sample_affine_problems(dimension=args.dimension,sample_count_dimension_factor=args.sample_count_dimension_factor,alphas=affine_alphas,max_instance_id=affine_max_instance_id, sample_range=sample_range)
sample_M4_problems(args.dimension,sample_count_dimension_factor=args.sample_count_dimension_factor,n_functions=args.m4_problem_count, sample_range=sample_range)
sample_bbob_problems(args.dimension,sample_count_dimension_factor=args.sample_count_dimension_factor, sample_range=sample_range)
#sample_cec_problems(args.dimension,sample_count_dimension_factor=args.sample_count_dimension_factor, sample_range=sample_range)