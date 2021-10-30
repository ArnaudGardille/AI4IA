import argparse
import json
import os
from importlib import import_module

import numpy as np

# from magnetic_bearing.hackathon.example_mock_model_submission.my_model import MyModel
# from magnetic_bearing.hackathon.example_gru_model_submission.my_model import MyModel
try:
    from utility_functions import load_data_csv
except:pass
try:
    from utilities.utility_functions import load_data_csv
except:pass
try:
    from sources.utilities.utility_functions import load_data_csv
except:pass

def save_hyper(hyper_rep,hyper_file,args):
    os.makedirs(hyper_rep,exist_ok=True)
    with open(os.path.join(hyper_rep,hyper_file), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
#################################################
# This script computes all the metrics necessary for the evaluation of
# the candidate's model performance. 
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
    
# Sagemaker API for fit (train + save)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    try:
        # Container environment
        parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
        parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
        parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    except:
       # Except
        print("Failed to load default container env keys. Using local default keys.")
        parser.add_argument('--model_dir', type=str, default='./models/')
        parser.add_argument('--data_dir', type=str, default='./data/')

    parser.add_argument('--model_def_file', type=str, default='my_model1')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--outputs_indexes',help='list of outputs indexes separated by comma (default: \"1,2,3,4,5\")', type=str, default="1,2,3,4,5")
    parser.add_argument('--hyper_fileName', type=str,default="hyper.json")
    
    # MANDATORY PARAMETERS FOR EVALUATION (METRICS COMPUTATION) -> DO NOT MODIFY THESE OPTIONS NAMES
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--train_fileName', type=str, default="input1.csv")
    parser.add_argument('--Ndecim', type=int,default=1)
    parser.add_argument('--outputs_prefix', type=str, default='output')
    
    args = parser.parse_args()
    
    save_hyper(args.model_dir,args.hyper_fileName,args)
    
    names_outputs = [args.outputs_prefix + k for k in args.outputs_indexes.split(",")]
    
    # Get Model Definition 
    try:
        MyModel = import_module('utilities.' + args.model_def_file).MyModel
    except:
        MyModel = import_module('sources.utilities.' + args.model_def_file).MyModel
        
    
    # And instanciate
    model = MyModel.create_model(gpu_available=args.use_gpu,
                                 epochs=args.epochs,lr=args.lr,
                                 output_size=len(names_outputs))
    
    # Load training Data
    ts, xs, ys = load_data_csv(data_path = args.data_dir + "/" + args.train_fileName,
                               name_time = "Time",
                               name_input= "input",
                               names_outputs = names_outputs,
                               Ndecim = args.Ndecim)
    
    # Train
    model.fit(xs=[xs], ys=[ys[:,k] for k in range(len(names_outputs))])
    
    # Save Trained Model
    model.save(args.model_dir)
