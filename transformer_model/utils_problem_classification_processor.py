
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import sklearn 
import seaborn as sns
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.dummy import DummyClassifier
from .utils import *
class ProblemClassificationProcessor():
    verbose:bool
    splitter:SplitterBase
    fold:int    
    task_name='problem_classification'
    def __init__(self, splitter=None, verbose=False,  fold=None, split_ids_dir=None, id_names=None):
   
        self.verbose=verbose
        self.fold=fold
        self.id_names=id_names
        self.splitter=splitter if splitter is not None else SplitterBBOBInstance()
    
    def get_x_y(self, sample_df, split_name, shuffle=True):
        print(sample_df)
        if self.verbose:
            print('Extracting x and y')
        xys=[]
        if 'problem_id' in self.id_names:
            problem_ids = sample_df.index.get_level_values('problem_id').drop_duplicates().sort_values().values
            instance_ids = sample_df.index.get_level_values('instance_id').drop_duplicates().sort_values().values
            for problem_id in problem_ids:
                problem_samples=sample_df.query("problem_id==@problem_id")
                for instance_id in instance_ids:
                    problem_instance_samples=problem_samples.query("instance_id==@instance_id")
                    xys+=[(problem_instance_samples.values, problem_id, (problem_id, instance_id))]
                    
        else:
            ids = sample_df.index.drop_duplicates().sort_values().values
            
            for f in tqdm(ids):
                problem_samples=sample_df.loc[f]
                columns_of_interest=list(filter(lambda x:x.startswith("x_"), problem_samples.columns))+['y']
                xys+=[(problem_samples[columns_of_interest].values, problem_samples['problem_id'].values[0], f)]
        if shuffle:
            random.shuffle(xys)
        x=np.array([xy[0] for xy in xys])
        y=np.array([xy[1] for xy in xys])
        ids=np.array([xy[2] for xy in xys])

        return SplitData(x=x,y=y,ids=ids,name=split_name)
        

    
    def split_data(self, sample_df):
        '''if self.fold==None:
            if self.verbose:
                print('Splitting data')
            instance_ids = list(set(sample_df.index.get_level_values('instance_id').values))
            train_instance_ids, test_instance_ids = train_test_split(instance_ids, test_size=0.2)
            train_instance_ids, val_instance_ids = train_test_split(train_instance_ids, test_size=0.2)
        else:
            if self.verbose:
                print('Reading split data')
            max_instance_id=sample_df.index.get_level_values('instance_id').max()
            train_instance_ids, val_instance_ids, test_instance_ids = [list(pd.read_csv(f'folds/problem_classification_{max_instance_id}_instances/{split_name}_{self.fold}.csv',index_col=[0])['0'].values) for split_name in ['train','val','test']]
        train, val, test =[sample_df.query(f"instance_id in @split_instance_ids").sample(frac=1) for split_instance_ids in [train_instance_ids, val_instance_ids, test_instance_ids]]'''
        
        if self.fold==None:
            if self.verbose:
                print('Splitting data')
            return self.splitter.split(sample_df)
        else:
            if self.verbose:
                print('Reading split data')
            sample_df['instance_id']=[int(f.split('_')[1]) for f in sample_df.index]
            max_instance_id=sample_df['instance_id'].max()
            train_instance_ids, val_instance_ids, test_instance_ids = [list(pd.read_csv(f'folds/problem_classification_{max_instance_id}_instances/{split_name}_{self.fold}.csv',index_col=[0])['0'].values) for split_name in ['train','val','test']]
    
        train, val, test =[sample_df.query(f"instance_id in @split_instance_ids").sample(frac=1).drop(columns=['instance_id']) for split_instance_ids in [train_instance_ids, val_instance_ids, test_instance_ids]]

        return [('train', train), ('val',val), ('test',test)]

    def offset_problem_id(self, sample_df):
        if 'problem_id' not in self.id_names:
            sample_df['problem_id']=[int(f.split('_')[0]) for f in sample_df.index]
            problem_ids = sample_df['problem_id'].values
        else:
            problem_ids = sample_df.index.get_level_values('problem_id').values
        unique_problem_ids=np.unique(problem_ids)
        name_to_index={problem_id: problem_id_index for problem_id_index, problem_id in enumerate(list(unique_problem_ids))}
        index_to_name={problem_id_index: problem_id for problem_id, problem_id_index in name_to_index.items()}

        new_problem_ids = [name_to_index[p] for p in problem_ids]
        index_names=list(sample_df.index.names)
        sample_df = sample_df.reset_index()
        sample_df['problem_id']=new_problem_ids
        sample_df = sample_df.set_index(index_names)
        return sample_df, name_to_index,  index_to_name

    
    
    def run(self,sample_df,train_seeds,val_seeds, id_names):
        sample_df, self.name_to_index,  self.index_to_name = self.offset_problem_id(sample_df)
        split_data= self.split_data(sample_df)

        data = {split_name: self.get_x_y(split_data, split_name, shuffle=(split_name!='test')) for split_name, split_data in split_data}

        
        return data
    
    def evaluate_model(self, y,probas,ids,save_to_file=None):
        predicted_classes = [int(p.argmax()) for p in probas]
        if self.index_to_name is not None:
            y = [self.index_to_name[yy] for yy in y]
            predicted_classes = [self.index_to_name[yy] for yy in predicted_classes]
            
        report_df=pd.DataFrame(classification_report(y,predicted_classes, output_dict=True))
        report_df.to_csv(save_to_file + '_classification_report.csv')
        print(report_df)
        confusion_matrix_df=pd.DataFrame(confusion_matrix(y, predicted_classes))
        confusion_matrix_df.to_csv(save_to_file + '_confusion_matrix.csv')


        d=pd.DataFrame([y, predicted_classes]).T
        if len(self.id_names)>1:
            d.index= pd.MultiIndex.from_tuples([tuple(i) for i in ids], names=self.id_names )
        else:
            d.index=ids
        d.columns=['ys','predictions']
        d.to_csv(save_to_file + '_ys_predictions.csv')

        return {'report': report_df, 'confusion': confusion_matrix_df}

    def train_dummy(self,y):
        
        self.dummy_model= DummyClassifier()
        self.dummy_model.fit(y.index,y)
        
    def evaluate_dummy(self, y_test, test_ids,save_to_file=None):
        #y_test=pd.DataFrame(y_test, index=test_ids, columns=self.performance_columns)

        y_pred=self.dummy_model.predict(test_ids)
        return self.evaluate_model(y_test, y_pred,test_ids,save_to_file)