import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

import gin
import models
from data import multitask, toy_data


@gin.configurable
def create_toy_dataset(
    num_tasks=10,
    num_data_per_task=500,
    noise_std=0.1,
    standardise=False):

    X, _, Y, _ = toy_data.create_data(
        num_tasks, num_data_per_task, noise_std=noise_std)
    
    dataset = multitask.Dataset(X=X, Y=Y, standardise=standardise)

    return dataset
    

@gin.configurable
def create_model(
    model_name,
    dataset,
    num_inducing=100,
    dim_latent=1,
    multi_output=False):

    dim_in = dataset.dim_in
    dim_out = dataset.dim_out
    num_latent = dataset.num_tasks

    if model_name == "SVGP":
        raise NotImplementedError()
    elif model_name == "MLGP":
        model = models.MLGP(
            dim_in, dim_out, dim_latent,
            num_latent, num_inducing,
            multi_output=multi_output)
    else:
        raise NotImplementedError()

    return model


