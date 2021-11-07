import argparse
import json
import os
import time

import torch
import numpy as np
from sklearn.metrics import mean_squared_error

from importlib import import_module

try:
    from utility_functions import load_data_csv
except:pass
try:
    from utilities.utility_functions import load_data_csv
except:pass
try:
    from sources.utilities.utility_functions import load_data_csv
except:pass

##################################################
# This script computes all the metrics necessary for the evaluation of
# the candidate's model performance. It can be called directly for a local evaluation
# or used as a sagemaker entry point for a remote computation.
# This is what we do in calc_metrics_sagemaker.py
# There is no need for the candidate to modify this script. 
# The model submitted must present the correct characteristics 
# so this script can be launched. 
# Please verify that all is ok before submission.
#
# See example notebook for an example of how to use this script
##################################################
## Author: François Caire
## Maintainer: François Caire
## Email: francois.caire at skf.com
##################################################

def save_metric(metric_filename, metric_dict: dict, description: dict, location='./'):
    filename = metric_filename
    os.makedirs(location, exist_ok=True)

    with open(os.path.join(location, filename), 'w') as f:
        for key in sorted(description):
            f.write('{} : {}\n'.format(key, description[key]))

        for key in sorted(metric_dict):
            f.write('{} : {}\n'.format(key, metric_dict[key]))

if __name__ == '__main__':

    # region args code
    parser = argparse.ArgumentParser()

    try:
        # Container environment
        parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
        parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
        parser.add_argument('--out_dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
        parser.add_argument('--model_dir',type=str,default='./')
    except:
        # Local 
        parser.add_argument('--data_dir', type=str,default='./data/DataSet_ex')
        parser.add_argument('--out_dir',type=str,default='./metrics')
        parser.add_argument('--model_dir',type=str,default='./models')
        
    parser.add_argument('--model_def_file',type=str,default='my_model1')
    parser.add_argument('--test_fileName', type=str, default='input2.csv')       
    parser.add_argument('--estimator_hyperParams_fileName', type=str, default='hyper.json')
    parser.add_argument('--model_kwargs_fileName', type=str, default='model_kwargs.json')

    args = parser.parse_args()

    print(os.path.join(args.model_dir,args.estimator_hyperParams_fileName))
    with open(os.path.join(args.model_dir,args.estimator_hyperParams_fileName),'r') as f:
        dHyper = json.load(f)
    with open(os.path.join(args.model_dir,args.model_kwargs_fileName),'r') as f:
        dKwargs = json.load(f)

    # endregion args code
    # CAUTION: JUST FOR TEST:
    if not torch.cuda.is_available() : dKwargs['device'] = 'cpu'
        
    # Get Model Definition 
    try:
        MyModel = import_module('utilities.' + args.model_def_file).MyModel
    except:
        MyModel = import_module('sources.utilities.' + args.model_def_file).MyModel
        
    # Instantiate model:
    model = MyModel(**dKwargs)

    # region fit
    train_filePath = os.path.join(args.data_dir,dHyper['train_fileName'])
    _ , x_train, y_train = load_data_csv(data_path=train_filePath,Ndecim=dHyper['Ndecim'])

    ts = time.time()
    model.fit(xs=[x_train], ys=[y_train[:,k] for k in range(5)])
    metric_train_time = time.time() - ts

    # endregion fit

    # region predict all
    _ , x_test , y_test = load_data_csv(data_path=os.path.join(args.data_dir,args.test_fileName))

    # Inference Time computation :
    ts = time.time()
    y_pred = model.predict_timeseries(x_test)
    inf_time =(time.time() - ts)

    metric_mean_inference_time  = inf_time/len(x_test)

    # Normalized Error Computation : 
    y_pred = model.predict_timeseries(x_test)
    max_ytest_value = np.zeros( (5,1) )
    for k in range(5) : # PLEASE NOTE THAT WE ONLY CONSIDER MODELS THAT COMPUTES ALL THE 5 OUTPUTS
        max_ytest_value[k] = np.max(np.abs(y_test[:,k]))
        
    metric_normalized_mse = np.zeros( (5,) )
    for k in range(5):
        metric_normalized_mse[k] = mean_squared_error(y_true=y_test[:,k]/max_ytest_value[k],y_pred=y_pred[:,k]/max_ytest_value[k])

    weights_mse = np.array( [1,1,1,1,1] )
    metric_normalized_mse_sum = np.dot(weights_mse,metric_normalized_mse)

    # End normalized error Computation
    
    aggregated_performance_indicator = metric_train_time/3600/1e3 + \
                                       metric_mean_inference_time + \
                                       metric_normalized_mse_sum  + \
                                       len(x_train)/1e6

    metrics_info = { 
                     'Training set size': len(x_train), 
                     'Training time [seconds]' : metric_train_time,
                     'Average inference time [seconds]' : metric_mean_inference_time,
                     'Normalized MSE' : metric_normalized_mse,
                     'Normalized MSE Sum' : metric_normalized_mse_sum,
                     'Aggregated Metrics': aggregated_performance_indicator 
                   }

    save_metric(metric_filename='{}_{}_metrics'.format(model.description['team_name'],model.description['model_name']),
                location=args.out_dir,
                metric_dict=metrics_info,
                description=model.description)


