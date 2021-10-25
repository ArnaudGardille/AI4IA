import os
import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# IMPORT THE MODEL API FROM WHICH YOUR MODEL MUST INHERITATE : 
try:
    from utilities.model_api import ModelApi
except:
    from sources.utilities.model_api import ModelApi
    
##################################################
## In this script, candidates are asked to implement two things:
#    1- the model, along with its training, prediction/inference methods;
#    2- the interface of this model with SKF evaluation and remote (AWS sagemaker) computation tools.
#
# See example notebook for an example of how to use this script
##################################################
## Author: François Caire
## Maintainer: François Caire
## Email: francois.caire at skf.com
##################################################

"""
BELOW IS THE IMPLEMENTATION OF YOUR MODEL

This example (GRUModel) implements a GRU class model inherited from a Pytorch model class.
Your solution can be any, the interface with higher level scripts needed (sagemaker_api and calc_metrics) 
as well as test_submission, is done in the class definition below (cf. MyModel def).

To conclude: the only constraint you have here is that your model is instanciated inside the MyModel class definition
where you have to implement all the methods required for the higher levels tools we need (and that you also need !) 
to evaluate your solution in terms of precision, inference and training times etc...
"""
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=85, nlayers=2,
                 device='cpu', lr=0.001, seed=0, epochs=25, output_size=5):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.nlayers = nlayers
        self.output_size = output_size
        self.gru = nn.GRU(input_size, self.hidden_layer_size, self.nlayers).to(device)
        self.hidden = torch.zeros(nlayers, 1, self.hidden_layer_size).to(device)
        self.linear = nn.Linear(self.hidden_layer_size, output_size).to(device)
        self.relu = nn.ReLU()
        self = self.to(device)
        self.device = device
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        
    def forward(self, input_vec):
        Ni = input_vec.shape[0]
        if isinstance(input_vec,np.ndarray):
            input_vec = torch.from_numpy(input_vec).float().to(self.device)

        output = torch.zeros((Ni,self.output_size)).to(self.device)
        h = self.hidden
    
        gru_out, _ = self.gru(input_vec.view(Ni,1,-1), h)
        predictions = self.linear(self.relu(gru_out))
        output = predictions.view(Ni,self.output_size)
        
        return output

    def fit(self, xs, ys):
        use_cuda = True if self.device == 'cuda' else False

        # set the seed for generating random numbers
        #torch.manual_seed(self.seed)
        #if use_cuda:55f3bogk
         #   torch.cuda.manual_seed(self.seed)
            
        self.output_shape = (xs.shape[0],len(ys))
        
        x = torch.tensor(xs,dtype=float)
        y = torch.tensor(ys,dtype=float).view(x.shape[0],-1)        

        seq_data = TensorDataset(x,y)
        batch_size = len(x)//5
        data_loader = DataLoader(seq_data, shuffle=False, batch_size=batch_size, drop_last=True)
        train_seq = [(i.clone().float().to(self.device), o.clone().float().to(self.device)) for i, o in data_loader]

        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        loss_function = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.95, patience=2, threshold=1e-3,
                                                               threshold_mode='rel', cooldown=0, min_lr=1e-4,
                                                               eps=1e-05, verbose=False)
        
        for i in range(self.epochs):
            self.train()
            # torch.cuda.empty_cache()
            for input, target in train_seq:
                optimizer.zero_grad()
                y_pred = self(input)
                single_loss = loss_function(y_pred, target)
                single_loss.backward()
                optimizer.step()
                scheduler.step(single_loss)
            
            if i%(self.epochs//25) == 0: print("it. %i / %i - loss = %.8f"%(i,self.epochs,single_loss))


"""
BELOW IS THE IMPLEMENTATION OF YOUR MODEL'S INTERFACE

Here you have to implement all the necessary methods as they are 
defined in the parent class ModelApi (cf file model_api.py).
These methods are used in higher levels scripts such as:
    - sagemaker_api.py that allows you (and us) to run training tasks on local or Amazon specific instances ;
    - calc_metrics.py/calc_metrics_on_sagemaker.py that allows you (and us) to compute the performance metrics of your solution, given your model definition (this file) and a test dataset as input;
"""
class MyModel(ModelApi):

    def __init__(self, **model_kwargs):
        self.model_kwargs = model_kwargs
        
        self.nn_model: GRUModel = GRUModel(**model_kwargs)

    def fit(self, xs: List[np.ndarray], ys: List[List[np.ndarray]], timeout=36000):
        self.nn_model.fit(xs, ys)

    @classmethod
    def get_sagemaker_estimator_class(self):
        """
        return the class with which to initiate an instance on sagemaker:
        e.g. SKLearn, PyTorch, TensorFlow, etc.
        by default - use SKLearn image.
        """
        
        from sagemaker.pytorch import PyTorch
        framework_version = '1.8.0'
        
        return PyTorch,framework_version

    def predict_one_timepoint(self, x: float) -> np.ndarray:
        return self.nn_model(np.asarray([x])).detach().numpy()

    def predict_timeseries(self, x: np.ndarray) -> np.ndarray:
        return self.nn_model(x).detach().numpy().T

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'w') as f:
            json.dump(self.model_kwargs, f)

        path = os.path.join(model_dir, 'model.pth')
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        torch.save(self.nn_model.cpu().state_dict(), path)

    @classmethod
    def load(cls, model_dir: str):
        path = os.path.join(model_dir, 'model_kwargs.json')
        with open(path, 'r') as f:
            model_kwargs = json.load(f)
            
        model_kwargs['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        my_model = cls(**model_kwargs)

        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            my_model.nn_model.load_state_dict(torch.load(f))

        return my_model

    @classmethod
    def create_model(cls, gpu_available: bool = False, **kwargs):
        return cls(device='cuda' if gpu_available and torch.cuda.is_available() else 'cpu', **kwargs)

    @property
    def description(self):
        team_name = 's2m'
        email = 'francois.caire@skf.com'
        model_name = 'GRU2'
        affiliation = 'SKF Group'
        description = 'This is a simple GRU model that supports 1 input and 1 to 5 outputs'
        technology_stack = 'pytorch'
        other_remarks = ''

        return dict(team_name=team_name,
                    email=email,
                    model_name=model_name,
                    description=description,
                    technology_stack=technology_stack,
                    other_remarks=other_remarks,
                    affiliation=affiliation)
