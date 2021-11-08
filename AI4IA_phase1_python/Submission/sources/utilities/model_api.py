# DO NOT MODIFY
# Given As-Is to candidates - do not modify
from typing import List

import numpy as np

##################################################
# This is the ModelApi virtual class definition
# It is mandatory that your model's interface (as in my_model1.py)  
# inheritates from this class
#
# There is no need for the candidate to modify this
##################################################
## Authors: Orion Talmi,François Caire
## Maintainer: François Caire
## Email: francois.caire at skf.com
##################################################


class ModelApi:
    """Interface for Magnetic Bearing Model. All the following methods,
    must be implemented by your model
    """

    @classmethod
    def create_model(cls, gpu_available:bool=False):
        """

        :param gpu_available: whether a gpu device is available.
        :return: an instance of your model. Use "return cls(*your_args, **your_kwargs)" to call your constructor.
        Note: create_model is expected to take no more than 30 seconds
        """
        raise NotImplementedError('model create_model() was not implemented')

    @property
    def description(self):
        """
        :return: a dictionary with the following properties
        """

        team_name = 'team_name'  #NO SPACE ALLOWED
        email = 'your_email@gmail.com'
        model_title = 'Model title - e.g. My Favorite Model'
        affiliation = 'Company/Instituition'
        description = 'description of the model and architecture'
        technology_stack = 'technology stack you are using, e.g. sklearn, pytorch'
        other_remarks = "put here anything else you'd like us to know"

        return dict(team_name=team_name,
                    email=email,
                    model_title=model_title,
                    description=description,
                    technology_stack=technology_stack,
                    other_remarks=other_remarks,
                    affiliation=affiliation)

    @classmethod
    def get_sagemaker_estimator_class(self):
        """
        return the class with which to initiate an instance on sagemaker:
        e.g. SKLearn, PyTorch, TensorFlow, etc.
        by default - use SKLearn image.

        """

        # Implementation examplea:
        """
        from sagemaker.sklearn import SKLearn
        return SKLearn
        """

        # or

        """
        from sagemaker.pytorch import PyTorch
        return PyTorch
        """

        raise NotImplementedError('model get_sagemaker_estimator_class() was not implemented')

    def fit(self, xs: List[np.ndarray], ys: List[np.ndarray], timeout=36000):
        """ train on several (x, y) examples
        :param xs: input data given as a list containing one dimensionnal array corresponding to input samples
        :param ys: output data given as a list of one dimensionnal arrays corresponding to output samples
        :param timeout: maximal time (on the hosting machine) allowed for this operation (in seconds).
        """

        raise NotImplementedError('model fit() was not implemented')

    def predict_timeseries(self, x: np.ndarray) -> np.ndarray:
        """ produce a prediction: x -> y where x is the entire time series from the beginning

        :param x: 1 input series given as a 2D ndarray with rows representing samples, and columns representing features.
        :return:  corresponding predictions as 2D np.ndarray

        Note: calling predict_series may change model's state.
        Note: self.predict_series(x) should return the same results as [self.predict_one_timepoint(xi) for xi in x] up to 5 digits precision.
        Note: predict_timeseries is expected to take no more than 1 second per sample
        """

        raise NotImplementedError('model predict() was not implemented')


    def save(self, model_dir:str):
        """ save the model to a file
        :param path: a path to the file which will store the model

        Note: save is expected to take no more than 10 minutes
        """

        raise NotImplementedError('model predict() was not implemented')

    @classmethod
    def load(cls, model_dir:str):#->ModelApi
        """ load a pretrained model from a file
        :param path: path to the file of the model.
        :return: an instance of this class initialized with loaded model.

        Note: save is expected to take no more than 10 minutes
        """

        raise NotImplementedError('model load() was not implemented')


    
