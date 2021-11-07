import os
import json
from re import S
from typing import List
import copy
import numpy as np
import time
import pandas as pd

from scipy import signal as sg
from scipy import integrate
from scipy.interpolate import interpn,interp1d,RegularGridInterpolator, Akima1DInterpolator
from scipy.optimize import minimize
import scipy.signal as sig
from scipy.optimize import least_squares


# IMPORT THE MODEL API FROM WHICH YOUR MODEL MUST INHERITATE : 
try:
    from model_api import ModelApi
except:pass
try:
    from utilities.model_api import ModelApi
except:pass
try:
    from sources.utilities.model_api import ModelApi
except:pass
    
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
class LTIModel():
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

def pred(x, t_e, entree):
    """
    Gives the response of a LTI system
    - at times t_e 
    - given an imput signal entree
    - given coefficients x
    """
    b = x[:int(len(x)/2)]
    a = x[int(len(x)/2):]
    H_1 = sig.lti(b,a)
    [t_s, sortie, xout] = H_1.output(entree,t_e)
    return np.array(sortie)

def err(x, t, entree, y):
    """ Computes the squared error between truth and prediction """
    return (pred(x, t, entree) - y)**2

def err_multi(x, times, entrees, outputs):
    """ Computes the squared error between truth and prediction at multiple times """
    error = 0
    for k in range(len(entrees)) :
        error += np.mean(err(x,times[k],entrees[k],outputs[k]))
    return error 

def RMSE(real,prediction):
    """ Computes the mean squared error between truth and prediction """
    rmse = 0
    for k in range(len(real)):
        rmse += (real[k]-prediction[k])**2
    return rmse/len(real)

def CalcOffsetShrink(x, y):
    ##Calcul des shrink (positif ou négatif) et des offset de chaque output

    offsets = []
    shrinks = []
    for i in range(len(y)):
        offset = y[i][0]
        offsets.append(offset)
        
        try:
            shrink = (y[i][-1]-offset[i])/x[-1]
        except:
            print("erreur calcul shrink")
            shrink = 1

        shrinks.append(shrink)
    
    return np.array(offsets), np.array(shrinks)

def removeOffsetShrink(ys, offsets, shrinks):
    ###Traitement des output
    ys = np.array(ys)
    ys_corrige = copy.deepcopy(ys)

    print(ys.shape)

        
    for i in range(ys.shape[0]):

        ys_corrige[i] -= offsets[i]
        ys_corrige[i] *= 1/shrinks[i]

    return ys_corrige

def addOffsetShrink(ys, offsets, shrinks):
    ###Traitement des output
    ys = np.array(ys)
    ys_corrige = copy.deepcopy(ys)

    print(ys.shape)

        
    for i in range(ys.shape[0]):
        ys_corrige[i] *= shrinks[i]
        ys_corrige[i] += offsets[i]
        

    return ys_corrige


"""
BELOW IS THE IMPLEMENTATION OF YOUR MODEL'S INTERFACE

Here you have to implement all the necessary methods as they are 
defined in the parent class ModelApi (cf file model_api.py).
These methods are used in higher levels scripts such as:
    - sagemaker_api.py that allows you (and us) to run training tasks on local or Amazon specific instances ;
    - calc_metrics.py/calc_metrics_on_sagemaker.py that allows you (and us) to compute the performance metrics of your solution, given your model definition (this file) and a test dataset as input;
"""
class MyModel(ModelApi):

    def __init__(self, degre=5, nbOutputs=5,  **model_kwargs):
        self.model_kwargs = model_kwargs
        self.degre = degre
        self.coeffs = np.ones(2*degre)
        self.nbOutputs = nbOutputs
        self.shrinks = np.ones(1)
        self.offsets = np.zeros(nbOutputs)
        

    def fit(self, xs: List[np.ndarray], ys: List[np.ndarray], timeout=36000, verbose=False):
        """
        Works with only one input
        """
        #inputs_train = xs #['input0','input1','input3','input5']
        #inputs_test = ys #['input2','input4','input6']
        print("xs", len(xs))
        print("ys", len(ys))


        x_train = np.array(xs[0])
        n = x_train.size


        times_train = np.arange(n)*0.001
        #params_output_2 = {}
        self.nbOutputs = np.array(ys).shape[0]
        Compteur = 0

        self.offsets, self.shrinks = CalcOffsetShrink(x_train, ys)

        #self.coeffs = np.ones(2*self.degre)
        #x = np.arange(1,2*self.degre+1)

        ys_corr = removeOffsetShrink(ys, self.offsets, self.shrinks)

        for y_train in ys_corr:
            res = least_squares(err_multi, self.coeffs, bounds=(-10, 10), args=(times_train,x_train,y_train), verbose=verbose, ftol=1e-15, gtol=1e-15, xtol = 1e-15, max_nfev = 50)
            self.coeffs = res.x

            Compteur += 1
            if verbose:
                print("Avancement : " + str(round(Compteur/N_total*100)) +"%")
                
            #params_output_2[name] = x

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

    def predict_timeseries(self, x: np.ndarray) -> np.ndarray:
        #print(x)
        #Sorties_v2 = copy.deepcopy(d)

        #for k in range(x.shape[0]):
        #print("shape: ",x[k].shape)
        outputs = np.zeros((self.nbOutputs, x.shape[-1]))
        Time = np.arange(x.size)*0.001
        for i in range(self.nbOutputs):
            outputs[i] = pred(self.coeffs,Time,x)*self.shrinks[i]+self.offsets[i]  ##On inverse la correction faite sur les outputs

        
        return outputs.T
        

        #return self.nn_model(x).detach().numpy()

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        filenames = ["degre", "coeffs", "nbOutputs", "shrinks", "offsets"]
    
        path = os.path.join(model_dir, "degre")
        np.save(path, self.degre)

        path = os.path.join(model_dir, "coeffs")
        np.save(path, self.coeffs)

        path = os.path.join(model_dir, "nbOutputs")
        np.save(path, self.nbOutputs)

        path = os.path.join(model_dir, "shrinks")
        np.save(path, self.shrinks)

        path = os.path.join(model_dir, "offsets")
        np.save(path, self.offsets)
            

    @classmethod
    def load(cls, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)


        model = cls() 

        path = os.path.join(model_dir, "degre.npy")
        print(path)
        model.degre = np.load(path)

        path = os.path.join(model_dir, "coeffs.npy")
        model.coeffs = np.load(path)

        path = os.path.join(model_dir, "nbOutputs.npy")
        model.nbOutputs = np.load(path)

        path = os.path.join(model_dir, "shrinks.npy")
        model.shrinks = np.load(path)

        path = os.path.join(model_dir, "offsets.npy")
        model.offsets = np.load(path)

        return model

    @classmethod
    def create_model(cls, gpu_available: bool = False, **kwargs):
        return cls(**kwargs)

    @property
    def description(self):
        team_name = 'MIA'
        email = 'arnaud.gardille@gmail.com'
        model_name = 'LTI'
        affiliation = 'Université Paris-Saclay'
        description = 'This is a simple LTI model that supports 1 input and 1 to 5 outputs'
        technology_stack = 'Scipy'
        other_remarks = ''

        return dict(team_name=team_name,
                    email=email,
                    model_name=model_name,
                    description=description,
                    technology_stack=technology_stack,
                    other_remarks=other_remarks,
                    affiliation=affiliation)
