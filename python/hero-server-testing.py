# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:30:19 2020

@author: rruiz26
"""

import gspread
import json

from oauth2client.service_account import ServiceAccountCredentials
import os
dir =  os.getcwd()
#from pprint import pprint  

with open("./db.json","r") as read_file:
    dict = json.load(read_file)

first_name = dict['users'][-1]['first name']
last_name = dict['users'][-1]['last name']
gender = dict['users'][-1]['gender']
user_id = str(dict['users'][-1]['messenger user id'])    

import numpy as np
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from sklearn.linear_model import RidgeCV
import yaml


pt = importr("policytree")
grf = importr("grf")
robj.numpy2ri.activate()

""" Functions """

def fit_policytree(xtrain, gammatrain, depth=2):
    pol = pt.policy_tree(X=xtrain, Gamma=gammatrain, depth=depth)
    return pol

def predict_policytree(pol, xtest, **kwargs):
    w = pt.predict_policy_tree(pol, np.atleast_2d(xtest), **kwargs)
    w = np.array(w, dtype=np.int_) - 1
    return w

def update_weighted_ridge_thompson(xs, yobs, ws, balwts, alpha, K, intercept = True):
    T, p = xs.shape

    """ Initialization """
    if intercept: p += 1
    A = np.empty((K, p, p))
    Ainv = np.empty((K, p, p))
    for k in range(K):
        A[k] = np.eye(p)
        Ainv[k] = np.eye(p)
    theta = np.zeros((K, p))
    sigma2 = np.empty((K, p, p))

    """ Construct weighting matrices """
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    W = [[] for _ in np.arange(K)]
    for t in range(T):
        for w in range(K):
            if ws[t] == w:
                W[w].append(balwts[t])
    W = np.array([np.diag(w) for w in W])

    """ Update """
    for w in range(K):
        idx = ws == w
        xw, yw = xs[idx], yobs[idx]
        xw, yw = standardization(xw), standardization(yw)
        if intercept: xw = np.hstack([np.ones((len(xw),1)), xw])
        A[w] = np.dot(np.transpose(xw) @ W[w], xw) + alpha * np.identity(p)
        Ainv[w]  = np.linalg.inv(A[w])
        theta[w] = Ainv[w] @ xw.T @ W[w] @ yw
        sigma2[w] = Ainv[w] * ((yw - xw @ theta[w]).T @ W[w] @ (yw - xw @ theta[w]))
    return theta, sigma2

def standardization(X, center = True, scale = True):
    X = np.array(X)
    if center: X = (X- X.mean(0))
    if scale: X = X / X.std(0)
    return X

def draw_weighted_ridge_thompson(xt, model, config, cache, intercept = True):
    """
    model should contain theta and sigma2 from the update_weighted_ridge_thompson function
    cache should contain past history
    """

    """ Config and initialization """
    if intercept:
        xt1 = np.empty(len(xt) + 1)
        xt1[0] = 1.0
        xt1[1:] = xt
    else:
        xt1 = np.array(xt) # ensure type
    K, num_mc, floor = config["K"], config["num_mc"], config["floor"]
    theta, sigma2 = model
    p = theta.shape[1]
    xs, ws = cache
    n = xs.shape[0]
    coeff = np.empty((K, num_mc, p))
    draws = np.empty((K, num_mc))

    for w in range(K):
        coeff[w] = np.random.multivariate_normal(theta[w], sigma2[w], size=num_mc, tol = 1e-6)
        draws[w] = coeff[w] @ xt1
    # calculate proportion of max and draw according to the probability
    ps = np.bincount(np.argmax(draws, axis=0), minlength=K) / num_mc
    ps = apply_floor(ps, floor)
    w = np.random.choice(K, p=ps)
    return w, ps

def apply_floor(a, amin):
    new = np.maximum(a, amin)
    total_slack = np.sum(new) - 1
    individual_slack = new - amin
    c = total_slack / np.sum(individual_slack)
    return new - c * individual_slack


def collect(a, indices):
    assert len(a) == len(indices)
    rows = np.arange(len(a))
    if np.ndim(indices) == 1:
       out = a[rows, indices]
    else:
       out = np.column_stack([a[rows, i] for i in indices.T])
    return out


def fit_ridge_lambda(xs, yobs, alpha = None):
    """
    This function is used to search for
    the best penalization parameter lambda
    given a set of observations
    """
    if alpha is not None:
        ridge = RidgeCV(alpha=alpha)
    else:
        ridge = RidgeCV(np.logspace(-6, 6, 13))
    ridge.fit(xs, yobs)
    lambda_min = ridge.alpha_
    return lambda_min


def forest_muhat_lfo(xs, ws, yobs, K, chunks, sample_weights=None, local_linear=False, **kwargs):
    """
    Fits a sequence of grf::regression_forests sequentially, ensuring that
    each prediction is only using past information. To be used for constructing
    doubly-robust scores in an adaptive experiment.
    Fitting and prediction are made in "chunks", so that predictions for
    the bth chunk are computed using the first b-1 chunks. Size of chunks
    is fixed and governed by 'chunks' argument. For the first 'chunk', all
    rows are zero.
    Chunks need not to correspond to batches in an adaptive experiment.
    """
    T = len(xs)
    if isinstance(chunks, int):
        timepoints = np.arange(start=chunks, stop=T, step=chunks).tolist() + [T]
    else:
        timepoints = chunks

    ws = ws.reshape(-1, 1)
    yobs = yobs.reshape(-1, 1)
    t0 = timepoints[0]
    muhat = np.empty((T, K))
    forest = fit_multi_forest(xs[:t0], ws[:t0], yobs[:t0], K=K,
                              sample_weights=sample_weights,
                              local_linear=local_linear,
                              **kwargs)
    muhat[:t0] = predict_multi_forest_oob(forest)
    for t0, t1 in zip(timepoints, timepoints[1:]):
        forest = fit_multi_forest(xs[:t1], ws[:t1], yobs[:t1], K=K,
                                  sample_weights=sample_weights,
                                  local_linear=local_linear,
                                  **kwargs)
        muhat[t0:t1] = predict_multi_forest_oob(forest)[t0:t1]
    return muhat


""" Utilities """


def fit_multi_forest(xs, ws, yobs, K,
                     sample_weights=None,
                     compute_oob_predictions=True,
                     local_linear=False,
                     **kwargs):
    """
    Fits K grf::regression_forests on data. When compute_oob_predictions is True,
        cross-fitting is ensured via OOB as follows.
    For each arm w:
        * forest = regression_forest(xs[ws == w], yobs[ws == w])
        * muhat[ws == w] = oob_predictions(forest)
        * muhat[ws != w] = predictions(forest, xs[ws != w])
    Note: if you are constructing doubly-robust scores for adaptively collected data, use forest_muhat_lfo instead.
    Example
    >>> T, K, p = 1000, 4, 3
    >>> xs = np.random.uniform(size=(T, p))
    >>> ws = np.random.randint(K, size=T)
    >>> yobs = np.random.uniform(size=T)
    >>> balwts = np.random.uniform(1, 10, size=T)
    >>> forest = fit_multi_forest(xs, ws, yobs, K, sample_weights=balwts, local_linear=True)
    >>> pred_oob, stderr_oob = predict_multi_forest_oob(forest, return_stderr=True)
    >>> pred, stderr = predict_multi_forest(forest, xs, return_stderr=True)
    """
    T = len(xs)
    ws = ws.reshape(-1, 1)
    yobs = yobs.reshape(-1, 1)
    forests = [None] * K
    assert np.issubdtype(ws.dtype, np.integer)

    for w in range(K):
        widx = np.where(ws == w)[0]
        sw = sample_weights[widx] if sample_weights is not None else robj.NULL
        forests[w] = fr = grf.regression_forest(
            X=xs[widx], Y=yobs[widx],
            compute_oob_predictions=compute_oob_predictions,
            sample_weights=sw,
            **kwargs)

        # Keep these indices it the forest is used for cross-fitting.
        if compute_oob_predictions:
            oidx = np.where(ws != w)[0]
            forests[w].widx = widx
            forests[w].oidx = oidx

            if local_linear:
                wpred = grf.predict_ll_regression_forest(fr, estimate_variance=True, **kwargs)
                opred = grf.predict_ll_regression_forest(fr, xs[oidx], estimate_variance=True, **kwargs)
            else:
                wpred = grf.predict_regression_forest(fr, estimate_variance=True, **kwargs)
                opred = grf.predict_regression_forest(fr, xs[oidx], estimate_variance=True, **kwargs)

            forests[w].oob_predictions = np.empty(T)
            forests[w].oob_predictions[widx] = wpred.rx2('predictions')
            forests[w].oob_predictions[oidx] = opred.rx2('predictions')

            forests[w].oob_stderr = np.empty(T)
            forests[w].oob_stderr[widx] = np.sqrt(wpred.rx2('variance.estimates'))
            forests[w].oob_stderr[oidx] = np.sqrt(opred.rx2('variance.estimates'))

            forests[w].local_linear = local_linear
    return forests


def predict_multi_forest_oob(forests, return_stderr=False):
    """ Retrieves the oob predictions and standard errors """
    if not hasattr(forests[0], 'oob_predictions'):
        raise ValueError("multi_forest was not fit with compute_oob_predictions=True.")
    predictions = np.column_stack([fr.oob_predictions for fr in forests])
    if return_stderr:
        variances = np.column_stack([fr.oob_stderr for fr in forests])
        return predictions, variances
    else:
        return predictions


def predict_multi_forest(forests, xs, return_stderr=False, **kwargs):
    """
    Predicts a list of forest fit using function fit_multi_forest.
    Note these predictions are NOT oob. Use predict_multi_forest_oob instead.
    """
    T = len(xs)
    K = len(forests)
    prediction_func = grf.predict_ll_regression_forest if forests[0].local_linear else grf.predict_regression_forest
    preds = [prediction_func(fr, xs, estimate_variance=return_stderr, **kwargs) for fr in forests]
    muhat = np.column_stack([p.rx2('predictions') for p in preds])
    if return_stderr:
        stderr = np.column_stack([np.sqrt(p.rx2('variance.estimates')) for p in preds])
        return muhat, stderr
    else:
        return muhat


def aw_scores(yobs, ws, balwts, K, muhat=None):
    scores = expand(balwts * yobs, ws, K)  # Y[t]*W[t]/e[t] term
    if muhat is not None:  # (1 - W[t]/e[t])*mu[t,w] term
        scores += (1 - expand(balwts, ws, K)) * muhat
    return scores


def expand(values, idx, num_cols):
    out = np.zeros((len(idx), num_cols), dtype=values.dtype)
    for i, (j, v) in enumerate(zip(idx, values)):
        out[i, j] = v
    return out
    
 
""" Experiment configuration """


basic_config = yaml.load(open('./python/liverun_config.yaml', "r"), Loader=yaml.FullLoader)
save = True
K = 40
p = 15 # TODO update
T = basic_config["T"]
num_batches = basic_config["num_batches"]
floor = basic_config["floor"]
policy = basic_config["policy_choice"]
ctr = basic_config['ctr']
nctr = [x for x in range(K) if x != ctr]
num_threads = basic_config['num_threads']

config = {
    **basic_config,
    "num_init_draws": max(int(eval(basic_config["num_init_draws"])), K*2) ,
    "final_batch": int(T - eval(basic_config['final_batch_list'])),
}

num_init_draws = config["num_init_draws"]
tlast = config["final_batch"]




# TODO: READ IN DATA HERE
# xt, a vector of numeric covariates; t indexed observation number
xt = np.random.normal(scale=1, size=p) # TODO change this to actual covariates that are read in
t = 0 # TODO agument this with every observation, potentially as a part of bandit model object
# xs, yobs, ws, probs: historical observations
xs = np.random.normal(scale=1, size=(2000, p))
yobs = np.random.normal(scale=1, size=2000)
ws = np.resize(range(40), 2000)
probs = np.full((2000, K), 1/K)

# placeholder model
model = (np.zeros((K, p+1)), np.repeat(np.identity(p+1)[np.newaxis, :, :], 40, axis=0))



# TODO update bandit model based on observed information
bandit_model = config['bandit_model']

update_times = np.linspace(num_init_draws - 1, tlast, num_batches + 1).astype(int)
last = slice(tlast, T)

if t in range(tlast):
    if t < num_init_draws:
        w = t % K
        p = 1 / K
    else:
        w, p = draw_weighted_ridge_thompson(xt, model, config, (xs, ws))
        control = np.random.binomial(1, p = 1/5)
        if control:
            w = ctr
        p[ctr] = 1 / 5 + p[ctr] * 4 / 5
        p[nctr] = p[nctr] * 4 / 5


# TODO Read in response; save all of these
yt = np.random.normal(scale=1, size=1)
xs_t = np.vstack((xs, xt))
yobs_t = np.concatenate((yobs, yt))
ws_t = np.concatenate((ws, [w]))
probs_t = np.vstack((probs, p))
balwts = 1 / collect(probs_t, ws_t)

if t in update_times:
    lambda_min = fit_ridge_lambda(xs_t, yobs_t)

    model = update_weighted_ridge_thompson(xs_t, yobs_t, ws_t, balwts, lambda_min, K,
                                           intercept=True)
    #TODO SAVE Model, or do every time?


timepoints = update_times + 1
timepoints = np.append(timepoints, T)
# Estimate muhat on first split
if t == tlast :
    muhat = forest_muhat_lfo(xs_t, ws_t, yobs_t, K, timepoints[:-1], num_threads=1) #TODO save, break into pieces
    aipw_scores = aw_scores(yobs_t, ws_t, balwts, K=K, muhat=muhat)
    bandit_model = fit_policytree(xs_t, aipw_scores) # TODO only read in once
    # TODO save below
    # ws[last] = predict_policytree(bandit_model, xs[last], num_threads=num_threads)
    # probs[np.arange(tlast, T), ws[last]] = 1
    # yobs[last] = collect(ys[last], ws[last])

# TODO save
data = dict(yobs=yobs_t, ws=ws_t, xs=xs_t, ys=yobs_t, probs=probs_t, final_bandit_model=bandit_model)





# output treatment to google spreadsheet 

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name(dir +"/python/creds.json", scope)

client = gspread.authorize(creds)

sheet = client.open("Testing").sheet1

insertRow = [user_id,first_name,last_name,gender,"This came from Heroku Server"]

sheet.append_row(insertRow)

