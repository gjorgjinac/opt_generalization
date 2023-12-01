import numpy as np
import uuid
import json
import pandas as pd
from config import *
from functools import partial


def zigzag(xs, k, m, lmbd):
    valz = np.zeros(xs.shape)
    xs = np.abs(xs)
    xs = xs / k - np.floor(xs / k)
    ids = xs <= lmbd
    valz = 1 - m + ids * m * (xs / lmbd) + (1 - ids) * m * (1 - (xs - lmbd) / (1 - lmbd))
    return valz

def F1(x, k, m, lmbd, M=np.array([[0.700458523713805,-0.140792306141100], [-0.140792306141100,0.933823944135235]]), s = np.array([-17,59])):
    n, d = x.shape
    val = 0
    x = np.matmul(x + s, M) #matmul replaced with dot
    f = lambda x: 10*np.abs(np.sin(0.1*x)) + 3*10**(-9)*zigzag(x, k, m, lmbd)*np.abs((x-40)*(x-185)*x*(x+50)*(x+180))
    val = np.sum(f(x), axis=1)
    return val

def F4(x, k, m, lmbd, M=np.array([[0.969630678081746, 0.032261150951298], [0.032261150951298, 0.965729170262545]]), s = np.array([-26, 79])):
    n, d = x.shape
    val = np.zeros((n, 1))
    x = np.matmul(x + s, M) #matmul replaced with dot
    f = lambda x: zigzag(x, k, m, lmbd) * 3 * np.abs(np.log(np.abs(x) * 1000 + 1)) + 30 - 30 * np.abs(np.cos(x / (np.pi * 10)))
    for i in range(n):
        val[i, 0] = np.sum(f(f(x[i, :])))
    return val

def F2(x, k, m, lmbd, M = np.array([[0.999481402986611, -0.007602368169921], [-0.007602368169921, 0.888553153414160]]), s = np.array([-92, -53])):
    n, d = x.shape
    val = 0
    x = np.matmul(x + s, M) #matmul replaced with dot
    f = lambda x: 10*np.abs(np.sin(0.1*x)) + 3*10**(-9)*zigzag(x, k, m, lmbd)*np.abs((x-40)*(x-185)*x*(x+50)*(x+180))
    val = np.sum(f(f(x)), axis=1)
    return val

def F3(x, k, m, lmbd, M = np.array([[0.724494251590697,-0.158624293816072], [-0.158624293816072,0.908670992406059]]), s = np.array([-74,-76])):
    n, d = x.shape 
    x = np.matmul(x + s, M) #matmul replaced with dot
    f = lambda x: zigzag(x, k, m, lmbd) * 3 * (np.abs(np.log(np.abs(x) * 1000 + 1))) + 30 - 30 * np.abs(np.cos(x / (np.pi * 10)))
    val = np.zeros((n, 1))
    for i in range(n):
        val[i] = np.sum(f(x[i, :]))
    return val



def transform_numpy_to_lists(dictionary):
    transformed_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            transformed_dict[key] = value.tolist()
        else:
            transformed_dict[key] = value
    return transformed_dict

def init_m4_functions(function_count,dimension,seed=42):
    
    methods = []
    np.random.seed(seed=seed)
    for i in range(function_count):
        fun = np.random.randint(1, high=5, dtype=int)
        k = np.random.randint(1, high=30, dtype=int)
        m = np.random.random()
        lmbd = np.random.random()
        s = np.random.uniform(low=-5, high=5, size=dimension)
        M = np.random.rand(dimension,dimension)

        md = {1: partial(F1, k=k, m=m, lmbd=lmbd, s=s, M=M), 
              2: partial(F2, k=k, m=m, lmbd=lmbd, s=s,M=M), 
              3: partial(F3, k=k, m=m, lmbd=lmbd, s=s,M=M), 
              4: partial(F4, k=k, m=m, lmbd=lmbd, s=s,M=M)}
        methods.append((md[fun], f'M{fun}_k_{k}_m{m}_l_{lmbd}_id_{i}'))
    return methods

def save_m4_functions_parameters(function_ids, dimension):
    all_function_parameters=[]
    for f, function_id in function_ids:
        function_parameters = transform_numpy_to_lists(f.keywords)
        print(function_parameters)
        function_parameters['function']=str(f)
        function_parameters['function_id']=function_id
        all_function_parameters+=[function_parameters]
    pd.DataFrame(all_function_parameters).to_csv(f'{data_dir}/random_functions/m4_{dimension}d.csv')

    
def get_function_from_parameters(data):
    function_number=data['function_id'].split("_")[0]
    function_mappings={"M1":F1, "M2": F2, "M3": F3, "M4":F4}
    return partial(function_mappings[function_number],k=data['k'], m=data['m'],lmbd=data['lmbd'],M=eval(data['M']), s=eval(data['s']))