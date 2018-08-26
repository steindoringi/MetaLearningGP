import tensorflow as tf
import numpy as np
import gpflow
import utils
import data
import sys, argparse, os

#GPflow
from gpflow import kernels
#from gpflow import mean_functions
from gpflow_mod import likelihoods, mean_functions

from gpflow.models.gpr import GPR
from gpflow.models.svgp import SVGP
from models import MLSVGP

#Plotting
import pylab
import seaborn as sns
colors = sns.color_palette("husl", 10)

#Training / Testing
import training


parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--train", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--train_steps", default=1000, type=int)
parser.add_argument("--infer_steps", default=1000, type=int)
parser.add_argument("--learning_rate", default=1e-2, type=float)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--n_tasks", default=6, type=int)
parser.add_argument("--n_train_tasks", default=2, type=int)
parser.add_argument("--n_train_data", default=30, type=int)
parser.add_argument("--n_test_data", default=5, type=int)
parser.add_argument("--n_inducing", default=10, type=int)
parser.add_argument("--dim_in", default=1, type=int)
parser.add_argument("--dim_out", default=1, type=int)
parser.add_argument("--dim_h", default=1, type=int)
parser.add_argument("--max_lik_h", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--model_path", default="checkpoints/MLGP.ckpt", type=str)
parser.add_argument("--model_name", default="MLGP", type=str)


ARGS = parser.parse_args()
np.random.seed(ARGS.seed)

""" Generate toy data from a GP """

D, X, F, Y = utils.generate_toy_data(ARGS.seed, ARGS.n_tasks, ARGS.n_train_data)

seq = np.int32([1, 4, 0, 2, 3, 5])
X, F, Y = X[seq], F[seq], Y[seq]

for task in range(ARGS.n_tasks):
    pylab.plot(D, F[task])
    pylab.scatter(X[task], Y[task])
pylab.savefig("all_data.png")
pylab.clf()

dataset = data.MultiTaskData(
    X, Y, D, ARGS.n_train_data, ARGS.n_tasks, ARGS.n_train_tasks,
    ARGS.dim_in, ARGS.dim_out, normalize=True)

""" Train MLGP """

X_train = dataset.data["training"]["X"].reshape(-1, ARGS.dim_in)
Z = X_train.copy()
np.random.shuffle(Z)
Z = Z[:ARGS.n_inducing]
Z = np.hstack([Z, np.zeros_like(Z)])

#mean_func = None
mean_func = mean_functions.Linear(A=np.ones((ARGS.dim_in+ARGS.dim_h, ARGS.dim_out)), b=None, name=ARGS.model_name+"_linear_mean")
kernel = kernels.RBF(ARGS.dim_in+ARGS.dim_h, ARD=True, name=ARGS.model_name+"_kernel")
likelihood = likelihoods.MultiGaussian(dim=ARGS.dim_out, name=ARGS.model_name+"_likelihood")

# Create graph
model = MLSVGP(
    dim_in=ARGS.dim_in, dim_out=ARGS.dim_out, dim_h=ARGS.dim_h, num_h=ARGS.n_tasks,
    kern=kernel, likelihood=likelihood, mean_function=mean_func, Z=Z,
    max_lik_h=ARGS.max_lik_h, name=ARGS.model_name)

likelihood = model._build_likelihood()
objective = -likelihood

if ARGS.train:
    session, saver = training.train(model, objective, dataset, ARGS)
else:
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, ARGS.model_path)

""" Infer Task Variable for Test Tasks """

session, saver = training.infer_latent(
    model, objective, session, saver, dataset, ARGS)

""" Evaluate """

use_var = True if mean_func is None else False

X_pred = dataset.get_domain(n_tasks=ARGS.n_tasks)
ids = np.int32(list(range(ARGS.n_tasks)))

ymu, yvar, hmu, hvar = training.predict(
    model, session, X_pred, ids, use_var=use_var)

ymu = (ymu * dataset.Y_std) + dataset.Y_mu
yvar = (dataset.Y_std**2) * yvar
ymin = ymu - 2*np.sqrt(yvar)
ymax = ymu + 2*np.sqrt(yvar)

print("Latent Variables")
for t in range(ARGS.n_tasks):
    print(hmu[t])

for t in range(ARGS.n_train_tasks):
    x = D.reshape(-1)
    task = dataset.data["training"]["ids"][t]
    pylab.plot(x, F[task], color="black")
    pylab.plot(x, ymu[task], color=colors[task])
    pylab.scatter(X[task], Y[task], color=colors[task])
    pylab.fill_between(x, ymin[task].reshape(-1), ymax[task].reshape(-1), alpha=0.3, color=colors[task])

pylab.savefig("training_pred.png")
pylab.clf()

for t in range(ARGS.n_tasks - ARGS.n_train_tasks):
    x = D.reshape(-1)
    task = dataset.data["test"]["ids"][t]
    pylab.plot(x, F[task], color="black")
    pylab.plot(x, ymu[task], color=colors[task])
    pylab.scatter(X[task], Y[task], color=colors[task])
    pylab.fill_between(x, ymin[task].reshape(-1), ymax[task].reshape(-1), alpha=0.3, color=colors[task])

pylab.savefig("test_pred.png")
pylab.clf()
