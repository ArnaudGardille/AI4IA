import unittest
import sys
from pprint import pprint
from importlib import import_module

import numpy
import numpy as np

    
##################################################
# This script defines and launches unittests on a model definition file
# It is provided so the candidates can test at any time the validity of the model definition 
# he intends to submit.
# Indeed, the evaluation of the submission necessitates the metrics computation 
# on sagemaker instance which is done by launching the calc_metrics* scripts
#
# See example notebook for an example of how to use this script
##################################################
## Authors: Orion Talmi, François Caire
## Maintainer: François Caire
## Email: francois.caire at skf.com
##################################################
try:
    from utility_functions import load_data_csv
except:pass
try:
    from utilities.utility_functions import load_data_csv
except:pass
try:
    from sources.utilities.utility_functions import load_data_csv
except:pass

class TestSubmission(unittest.TestCase):

    def test_mymodel(self):
        with self.subTest(name='create test data'):
            data = './data/DataSet_ex/input1.csv'
            _,self.xs,self.ys = load_data_csv(data)

        with self.subTest(name='create model'):
            model = MyModel.create_model()

        with self.subTest(name='fit 1'):
            x_train,y_train = [self.xs[:50]],[self.ys[:50,k] for k in range(5)]
            model.fit(x_train,y_train)
            y_pred = model.predict_timeseries(self.xs[:50])
            self.assertTrue(isinstance(y_pred,np.ndarray))
            print("y_pred", y_pred.shape)
            self.assertEqual(len(y_pred),50)

        with self.subTest(name='print description'):
            description = model.description
            print("Model Description : ")
            pprint(description)
            self.assertTrue(isinstance(description, dict))

        with self.subTest(name='save/load 1'):
            model.save('./tmp/')
            model2 = model.load('./tmp/')
            self.assertEqual(model.description, model2.description)

        with self.subTest(name='predict all'):
            y2 = model2.predict_timeseries(self.xs[:50])
            self.assertTrue(isinstance(y2, np.ndarray))
            self.assertEqual(len(y2[0,:]), 5)
            self.assertEqual(y2[:,0].shape, self.xs[:50].shape)
            self.assertEqual(y2[:,0].shape, self.ys[:50,0].shape)

        with self.subTest(name='compare prediction after predictions'):
            y1 = model.predict_timeseries(self.xs[-50:-1])
            y2 = model2.predict_timeseries(self.xs[-50:-1])
            self.assertTrue(np.all(y1 == y2))

if __name__ == '__main__':
    
    try:
        name_model = sys.argv[1]
        try:
            MyModel = import_module('sources.utilities.' + name_model).MyModel
        except:pass
        try:
            MyModel = import_module('utilities.' + name_model).MyModel
        except:pass
        try:
            MyModel = import_module(name_model).MyModel
        except:pass
        sys.argv[1:] = []
    except:
        print("Error: your model definition module could not be imported")
    
    unittest.main()