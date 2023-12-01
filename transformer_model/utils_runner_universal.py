from .tsai_custom import *
from tsai.all import *
computer_setup()
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, MSELoss
from fastai.losses import BaseLoss
import sklearn 
from fastai.callback.tracker import EarlyStoppingCallback,SaveModelCallback
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from .utils import *
from .utils_ranking_metrics import *
from .model_stats import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import os
from .config import *
from fastai.metrics import mse
import colorcet as cc
import matplotlib.pyplot as plt

import matplotlib
matplotlib.cm.register_cmap("my_cmap", my_cmap)


    
    
class UniversalRunner():

    logger = None
    verbose: bool
    plot_training: bool
    n_heads:int 
    n_layers:int 
    d_model:int
    d_k:int
    d_v:int 
    n_epochs:int
    bs:int

    model:any
    learner:any
    result_dir:str
    use_positional_encoding:bool
    
    def __init__(self,data_processor,extra_info=None, verbose=True, lr_max=5e-4, plot_training=True, use_positional_encoding=False, n_heads=1, n_layers=1, d_model=20, d_k=10, d_v=10, n_epochs=100, batch_size=8, fold=None, iteration_count=None, include_iteration_in_x=False, train_seeds=None, val_seeds=None, global_result_dir='results', aggregations=None, id_names=None, split_type=None, dropout=0.1, fc_dropout=0.1):

        self.verbose = verbose
        self.lr_max=lr_max
        self.plot_training=plot_training
        self.n_heads=n_heads
        self.n_layers=n_layers
        self.d_model=d_model
        self.dropout=dropout
        self.fc_dropout=fc_dropout
        self.d_k=d_k
        self.d_v=d_v
        self.id_names=id_names
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.train_seeds=train_seeds
        self.val_seeds=val_seeds

        self.data_processor=data_processor
        self.extra_info=extra_info
        self.extra_info += f'_n_heads_{n_heads}_n_layers_{n_layers}_d_model_{d_model}_d_k_{d_k}_d_v_{d_v}_aggregations_{"all" if aggregations is None else "-".join(aggregations)}'

        self.result_dir = os.path.join(global_result_dir,self.data_processor.task_name,self.extra_info)
        self.fold=fold
        self.use_positional_encoding=use_positional_encoding
        self.iteration_count=iteration_count
        self.aggregations=aggregations
        os.makedirs(self.result_dir, exist_ok = True) 
        



    def find_learning_rate(self, dls):
        print('Determining learning rate')
        learn = Learner(dls, self.model, loss_func=LabelSmoothingCrossEntropyFlat(), metrics=[RocAuc(), accuracy],  cbs=ShowGraphCallback2())
        learn.lr_find()


    def train_model(self, dls):
        print('Training model')
        callbacks=[EarlyStoppingCallback( min_delta=0.01,patience=2), SaveModelCallback (monitor='valid_loss', comp=None, min_delta=0.001,
                    fname=self.extra_info, every_epoch=False, at_end=False, with_opt=False, reset_on_fit=True)]

        if self.plot_training:
            callbacks+=[ShowGraphCallback2()]
            
        if self.data_processor.task_name!='performance_prediction':
            self.learner = Learner(dls, self.model, loss_func=CrossEntropyLoss(), metrics=[ accuracy],  cbs=callbacks)
        else:
            self.learner = Learner(dls, self.model, loss_func=MSELoss(), metrics=[weighted_mse, mse, calculate_misrankings_score, calculate_loss], cbs=callbacks) #
            #Learner(dls, self.model, loss_func=calculate_loss_torch, metrics=[weighted_mse, mse, calculate_misrankings_score, calculate_loss], cbs=ShowGraphCallback2()).lr_find()
        start = time.time()
        self.learner.fit(self.n_epochs, lr=self.lr_max)
 
        print('\nElapsed time:', time.time() - start)
        if self.plot_training:
            self.learner.plot_metrics()


    def evaluate(self, split_data:SplitData ):
        print('Evaluating model')
        probas, targets, preds = self.learner.get_X_preds(np.swapaxes(split_data.x,1,2), with_decoded=True)


        if split_data.name=='train':
            print(split_data.y.mean())
            print('Training dummy')
            self.data_processor.train_dummy(pd.DataFrame(split_data.y,index=split_data.ids))

        dummy_scores = self.data_processor.evaluate_dummy(split_data.y,split_data.ids, os.path.join(self.result_dir,f'{split_data.name}_dummy'))
        transformer_scores = self.data_processor.evaluate_model( split_data.y,probas,split_data.ids, os.path.join(self.result_dir,f'{split_data.name}_transopt'))

        return {(split_data.name,'transformer'):transformer_scores, (split_data.name, 'dummy'):dummy_scores}



    def get_batch_embeddings(self, batch):
        batch_embeddings=self.model.cuda().get_embeddings(batch[0].cuda())
        batch_embeddings=batch_embeddings.detach().cpu().numpy()
        return batch_embeddings


    def get_embeddings_from_dls(self, dls, batch_count=None, cast_y_to_int=True):
        #batch = dls.one_batch()
        all_embeddings=None
        all_labels=[]
        i=0
        for batch in dls:
            batch_embeddings=self.get_batch_embeddings(batch)
            all_embeddings=batch_embeddings if all_embeddings is None else np.append(all_embeddings,batch_embeddings, axis=0)
            all_labels+=list(batch[1])
            i+=1
            if batch_count is not None and i >= batch_count:
                break
        if cast_y_to_int:
            all_labels=[int(i) for i in all_labels]
        return all_embeddings, all_labels

    def show_cosine_similarity(self,embeddings,labels, count=10):
        plt.figure(figsize=(10,10)) 
        if self.processor.index_to_name is not None:
            labels = [self.processor.index_to_name[yy] for yy in labels]
        embeddings, labels = embeddings[:count], labels[:count]
        similarity = cosine_similarity(embeddings, embeddings)
        labels_sorted=labels.copy()
        labels_sorted.sort()
        similarity_df = pd.DataFrame(similarity, index=labels, columns=labels).sort_index()[labels_sorted]
        sns.heatmap(similarity_df, cmap=my_cmap)
        plt.show()
        
    def plot_embeddings(self, data_loader, batch_count):
        
 
        embeddings, labels = self.get_embeddings_from_dls(data_loader, batch_count)
        print(embeddings.shape)
        print(labels)
        self.show_cosine_similarity(embeddings,labels,100)
        if self.processor.index_to_name is not None:
            labels = [self.processor.index_to_name[yy] for yy in labels]
        
        tsne=sklearn.manifold.TSNE(n_components=2)
        batch_embeddings_2d=pd.DataFrame(tsne.fit_transform(embeddings) , columns=['x','y'])
        batch_embeddings_2d['label']=labels
        plt.figure(figsize=(10,10)) 
        sns.scatterplot(batch_embeddings_2d, x='x',y='y', hue='label', style='label',  palette=sns.color_palette(cc.glasbey, n_colors=24))
        plt.show()
    
    def init_model(self,dls):
        self.model=OptTransStats(dls.vars, dls.c, dls.len, n_heads=self.n_heads, n_layers=self.n_layers, d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, use_positional_encoding=self.use_positional_encoding, iteration_count=self.iteration_count, aggregations=self.aggregations, dropout=self.dropout, fc_dropout=self.fc_dropout)
            
    
    def save_embeddings(self, sample_df, split_name):
        data=self.data_processor.get_x_y(sample_df, split_name, shuffle=False)
        dset = TSDatasets(np.swapaxes(data.x,1,2),data.y)
        dls = TSDataLoaders.from_dsets(dset, bs=24, shuffle=False)
        embeddings=self.get_embeddings_from_dls( dls[0], None, cast_y_to_int=False)


        embedding_df=pd.DataFrame(embeddings[0])
        embedding_df[self.id_names]=[tuple([int(cc) for cc in c.cpu()]) for c in embeddings[1]]
        embedding_df=embedding_df.set_index(self.id_names)
        embedding_df.to_csv(os.path.join(self.result_dir,f'embeddings.csv'), compression='zip')
        return embedding_df

    def run (self,sample_df, plot_embeddings=False, save_embeddings=True, regenerate=False):
            
        data = self.data_processor.run(sample_df, self.train_seeds, self.val_seeds, self.id_names)
        dsets = {split_name: TSDatasets(torch.tensor(np.swapaxes(split_data.x,1,2)),torch.tensor(split_data.y)) for split_name, split_data in data.items()}
        
        dls = TSDataLoaders.from_dsets(dsets['train'], dsets['val'], bs=self.batch_size)
        
        test_data_loader = TSDataLoaders.from_dsets(dsets['test'], bs=self.batch_size)[0]
        
        dls.c=len(set(data['train'].y)) if self.data_processor.task_name!='performance_prediction' else data['train'].y[0].shape[0]
        
        print(f'Number of samples: {dls.len}, Number of variables: {dls.vars}, Number of classes: {dls.c}')
        self.init_model(dls)
        if plot_embeddings:
            self.plot_embeddings(test_data_loader, 100 )
        self.train_model(dls)
        torch.save(self.model, os.path.join(self.result_dir,f'trained_model.pt'))
        torch.save(self.model.state_dict(), os.path.join(self.result_dir,f'trained_model_dict.pt'))

        
        if plot_embeddings:
            self.plot_embeddings(test_data_loader, 100 )
        
        embedding_df=None
        if save_embeddings:
            embedding_df=self.save_embeddings(sample_df,'all')
        #self.check_random_forest_performance(dset_train, dset_val, dset_test)
        
        
        return self.model, self.learner, embedding_df, [self.evaluate(split_data) for split_data in data.values()]
    
    def evaluate_other_test(self,sample_df,name):
        data= self.data_processor.process_test_only(sample_df,name)
        return self.evaluate(data )
    

        