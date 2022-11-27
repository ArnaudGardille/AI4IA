import argparse
import os
import boto3
import sagemaker
import tarfile
import json

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
# the candidate's model performance on a remote AWS instance. 
# It uses the calc_metrics.py script as an entry point.
# There is no need for the candidate to modify this script.
# This script will be used for the evaluation of the candidates 
# after they submitted their solution
#
# See example notebook for an example of how to use this script
##################################################
## Author: François Caire
## Maintainer: François Caire
## Email: francois.caire at skf.com
##################################################

def download_from_s3(prefix_out, desc, out_dir='./out'):
    s3 = boto3.resource('s3')
    filename = 'model.tar.gz'
    metric_file_path = prefix_out + '/' + desc['TrainingJobName'] + '/' + 'output' + '/' + filename

    os.makedirs(out_dir, exist_ok=True)
    local_filename = os.path.join(out_dir, filename)
    s3.Bucket(bucket).download_file(metric_file_path, local_filename)

    with tarfile.open(local_filename, 'r:gz') as archived:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archived, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_run', type=float, default=4*60*60) #Max run of the dedicated estimator on AWS instance. Note: the default value will be applied for evaluation 
    parser.add_argument('--model_def_file', type=str, default='my_model1')
    parser.add_argument('--model_dir', type=str, default='./models/model0')
    parser.add_argument('--data_dir', type=str, default='./data/DataSet_ex')
    parser.add_argument('--out_dir',type=str, default='./metrics')
    parser.add_argument('--test_fileName', type=str, default='input2.csv')        
    parser.add_argument('--estimator_hyperParams_fileName', type=str, default='hyper.json')
    parser.add_argument('--model_kwargs_fileName', type=str, default='model_kwargs.json')  
    parser.add_argument('--role',type=str)
    parser.add_argument('--type_instance',type=str,default='ml.p3.2xlarge') #ml.m4.xlarge','ml.p3.2xlarge', #'ml.m4.xlarge',#'ml.p2.xlarge',#'ml.p3.2xlarge',#
    
    args = parser.parse_args()
    
    # Get Model Definition 
    try:
        MyModel = import_module('utilities.' + args.model_def_file).MyModel
    except:
        MyModel = import_module('sources.utilities.' + args.model_def_file).MyModel
    
    sagemaker_session = sagemaker.Session()
    
    bucket = sagemaker_session.default_bucket()
    prefix_in  = 'DEMO-AI4IA/input'
    prefix_out = 'DEMO-AI4IA/metrics'
   
    ######################
    sagemaker_estimator,framework_version = MyModel.get_sagemaker_estimator_class()
    
    estimator = sagemaker_estimator ( entry_point = 'calc_metrics.py',
                                      source_dir  = 'sources',
                                      role = args.role,
                                      py_version = 'py3',
                                      max_run = args.max_run,
                                      framework_version = framework_version,
                                      instance_count = 1,
                                      instance_type=args.type_instance,
                                      output_path=f's3://'+bucket+'/'+prefix_out,
                                      hyperparameters={
                                          'model_kwargs_fileName' : args.model_kwargs_fileName,
                                          'estimator_hyperParams_fileName' : args.estimator_hyperParams_fileName,
                                          'test_fileName' : args.test_fileName,
                                          'model_def_file': args.model_def_file
                                      },
                                      dependencies=[ os.path.join(args.model_dir,args.estimator_hyperParams_fileName),\
                                                     os.path.join(args.model_dir,args.model_kwargs_fileName) ]
                                    )
    
    input_channel = sagemaker_session.upload_data(path=args.data_dir, bucket=bucket, key_prefix=prefix_in)
    
    estimator.fit({'training': input_channel})
    
    ######################
    
    training_job_name = estimator.latest_training_job.name
    desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
    
    download_from_s3(prefix_out, desc, out_dir=args.out_dir)