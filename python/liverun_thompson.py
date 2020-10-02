import gspread

from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import numpy as np
import rpy2.robjects as robj
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import yaml
from sklearn.linear_model import RidgeCV

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

dir =  os.getcwd()

creds = ServiceAccountCredentials.from_json_keyfile_name(dir +"/python/creds.json", scope)

client = gspread.authorize(creds)

sheet = client.open("Testing").sheet1

full_dataset = client.open("Testing").sheet2

pt = importr("policytree")
grf = importr("grf")
base = importr("base")
numpy2ri.activate()

""" Functions """


def fit_policytree(xtrain, gammatrain, depth=2):
    pol = pt.policy_tree(X=xtrain, Gamma=gammatrain, depth=depth)
    return pol


def predict_policytree(pol, xtest, **kwargs):
    w = pt.predict_policy_tree(pol, np.atleast_2d(xtest), **kwargs)
    w = np.array(w, dtype=np.int_) - 1
    return w


def update_weighted_ridge_thompson(xs, yobs, ws, balwts, alpha, K, intercept=True):
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
        if intercept: xw = np.hstack([np.ones((len(xw), 1)), xw])
        A[w] = np.dot(np.transpose(xw) @ W[w], xw) + alpha * np.identity(p)
        Ainv[w] = np.linalg.inv(A[w])
        theta[w] = Ainv[w] @ xw.T @ W[w] @ yw
        sigma2[w] = Ainv[w] * ((yw - xw @ theta[w]).T @ W[w] @ (yw - xw @ theta[w]))
    return theta, sigma2


def standardization(X, center=True, scale=True):
    X = np.array(X)
    if center: X = (X - X.mean(0))
    if scale: X = X / X.std(0)
    return X


def draw_weighted_ridge_thompson(xt, model, config, intercept=True, num_mc=1000):
    """
    model should contain theta and sigma2 from the update_weighted_ridge_thompson function
    """

    """ Config and initialization """
    if intercept:
        xt1 = np.empty(len(xt) + 1)
        xt1[0] = 1.0
        xt1[1:] = xt
    else:
        xt1 = np.array(xt)  # ensure type
    K, floor = config["K"], config["floor"]
    theta, sigma2 = model
    p = theta.shape[1]
    coeff = np.empty((K, num_mc, p))
    draws = np.empty((K, num_mc))

    for w in range(K):
        coeff[w] = np.random.multivariate_normal(theta[w], sigma2[w], size=num_mc, tol=1e-6)
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


def fit_ridge_lambda(xs, yobs, alpha=None):
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


# READ IN DATA FOR CURRENT OBSERVATION
    
with open("./db.json", "r") as read_file:
    dict = json.load(read_file)
input = dict['users'][-1]  

try:
    input['dv_send_post8']
    responded = 1
except:
    # then we compute the individals treatment 
    responded = 0

    

if responded == 0 :
    # Compute Treatment 
    
    # EXPERIMENT CONFIGURATION
    basic_config = yaml.load(open('liverun_config.yaml', "r"), Loader=yaml.FullLoader)
    K = basic_config["K"]
    p = basic_config["p"] # number of coverates - currently 80 
    T = basic_config["T"]
    num_batches = basic_config["num_batches"]
    floor = basic_config["floor"] / T
    ctr = basic_config['ctr']
    nctr = [x for x in range(K) if x != ctr]
    num_threads = basic_config['num_threads']
    
    config = {
        **basic_config,
        "num_init_draws": max(int(eval(basic_config["num_init_draws"])), K * 2),
        "final_batch": int(T - 1 - eval(basic_config['final_batch_list'])),
        "floor": floor
    }
    
    num_init_draws = config["num_init_draws"]
    tlast = config["final_batch"]
    update_times = np.linspace(num_init_draws - 1, tlast, num_batches + 1).astype(int)
    
# Retrieve covariates    
    
    # Gender identification
    if input['gender'] == 'male':
        male = 1
    else:
        male = 0
    
    # Age
    age = int(input['cv_age'])
    
    # TODO: rename remaining covariates according to chatfuel attributes, see Zap: https://zapier.com/app/editor/100129520/
    # Education
    ed = int(input['cv_education'])
    ed_flag = 1*(ed==0)
    
    
    # Urban/rural
    urban = 1*(input['urban_rural'] == 'mostly urban')
    
    # Religion
    rel_none = 1*(input['cv_religion'] == 'None')
    rel_christian = 1*(input['cv_religion'] == 'Christian')
    rel_muslim = 1*(input['cv_religion'] == 'Muslim')
    rel_traditionalist = 1*(input['cv_religion'] == 'Traditionalist')
    rel_other = 1 - (rel_none + rel_christian + rel_muslim + rel_traditionalist)
    # Christian denomination
    denom_pentecostal = 1*(input['cv_religion'] == 'Pentecostal')
    # Religiosity
    religiosity_flag = 0
    if input['cv_religion_freq'] == 'Never':
        religiosity = 1
    elif input['cv_religion_freq'] == '< Once a month':
         religiosity = 2
    elif input['cv_religion_freq'] == '1-3 times a month':
         religiosity = 3
    elif input['cv_religion_freq'] == 'Once a week':
         religiosity = 4
    elif input['cv_religion_freq'] == '> Once a week':
         religiosity = 5
    elif input['cv_religion_freq'] == 'Daily':
         religiosity = 6
    else:
        religiosity = 0
        religiosity_flag = 1
    
    # Belief in God's Control
    god = 1*(input['cv_gods_control'] == '1')
    
    #Locus of Control
    locus = int(input['cv_locus_of_control'])
    locus_flag = 1*(locus==0)
    
    # Index of Scientific questions
    science = 1*(input['cv_science1'] == '1') + 1*(input['cv_science2'] == '1')
    science_flag = 1 - (1*(input['cv_science1'] == '1') + 1*(input['cv_science1'] == '2'))*(1*(input['cv_science2'] == '1') + 1*(input['cv_science2'] == '2'))
    
    # Digital literacy index
    dli = (
         # First element of DLI
         ( int(input['digital_phishing']) + int(input['digital_hashtag']) + int(input['digital_jpg']) + int(input['digital_malware']) + int(input['digital_cache']) + int(input['digital_rss']) )
         # Second element of DLI, friends
         + 2*(input['digital_friends'] == 'Somewhat agree') + 4*(input['digital_friends'] == 'Somewhat disagree') + 6*(input['digital_friends'] == 'Strongly disagree')
         # Third element of DLI, life
         + 2*(input['digital_life'] == 'Somewhat disagree' ) + 4*(input['digital_life'] == 'Somewhat agree' ) + 6*(input['digital_life'] == 'Strongly agree')
         # Fourth element of DLI, work
         + 2*(input['digital_work'] == 'Somewhat disagree' ) + 4*(input['digital_work'] == 'Somewhat agree' ) + 6*(input['digital_work'] == 'Strongly agree'))
    
    # Cognitive reflection test
    crt = 1*(input['crt_1'] == '5') + 1*(input['crt_2'] == '5') + 1*(input['crt_3'] == '47')
    
    # household index
    hhi = 1
    hhi_flag = 0
    hhi = (
         1*(input['cv_radio'] == 'I/my household owns') + #
         1*(input['cv_tv'] == 'I/my household owns') + #
         1*(input['cv_moto'] == 'I/my household owns') + #
         1*(input['cv_computer'] == 'I/my household owns') + #
         1*(input['cv_bank'] == 'I/my household owns') + #
         1*(input['cv_phone'] == 'I/my household owns') + #
         1*(input['cv_bike'] == 'I/my household owns')
             )
    hhi_flag = (1-(
             (1*(input['cv_radio'] == 'I/my household owns') + 1*(input['cv_radio']=='Do not own'))* #
             (1*(input['cv_tv'] == 'I/my household owns') + 1*(input['cv_tv']=='Do not own'))* #
             (1*(input['cv_moto'] == 'I/my household owns') + 1*(input['cv_moto']=='Do not own'))* #
             (1*(input['cv_computer'] == 'I/my household owns') + 1*(input['cv_computer']=='Do not own'))* #
             (1*(input['cv_bank'] == 'I/my household owns') + 1*(input['cv_bank']=='Do not own'))* #
             (1*(input['cv_phone'] == 'I/my household owns') + 1*(input['cv_phone']=='Do not own'))* #
             (1*(input['cv_bike'] == 'I/my household owns') + 1*(input['cv_bike']=='Do not own')))
             )
    
    # Job with cash income
    cash = 1*(input['cv_job'] == 'Yes')
    
    # Job description
    occ_0 = 1 * (input['cv_job_desc'] == '0')
    occ_1 = 1 * (input['cv_job_desc'] == '1')
    occ_2 =  1 * (input['cv_job_desc'] == '2')
    occ_3 =  1 * (input['cv_job_desc'] == '3')
    occ_4 =  1 * (input['cv_job_desc'] == '4')
    occ_5 =  1 * (input['cv_job_desc'] == '5')
    occ_6 =  1 * (input['cv_job_desc'] == '6')
    occ_7 =  1 * (input['cv_job_desc'] == '7')
    occ_8 =  1 * (input['cv_job_desc'] == '8')
    occ_9 =  1 * (input['cv_job_desc'] == '9')
    occ_10 = 1 *  (input['cv_job_desc'] == '10')
    occ_11 = 1 *  (input['cv_job_desc'] == '11')
    occ_12 = 1 *  (input['cv_job_desc'] == '12')
    occ_13 = 1 *  (input['cv_job_desc'] == '13')
    
    # Number of people in household
    hh = int(input['cv_hhold'])
    hh_flag = 1*(hh==0)
    
    # Political party affiliation
    pol = 1*(input['party'] == '1: Jubilee') + 1*(input['party'] == '1: APC')
    
    # Concern regarding COVID-19
    cov_concern = 1*(input['cv_risk'] == 'Not at all worried') + 2*(input['cv_risk'] == 'Somewhat worried') + 3*(input['cv_risk'] == 'Very worried')
    cov_concern_flag = 1*(cov_concern == 0)
    
    # COVID-19 information
    cov_info = 1*(input['cov_tf_1'] == 'True') + 1*(input['cov_tf_2'] == 'True') + 1*(input['cov_tf_3'] == 'True')
    
    # Perceived government efficacy on COVID-19
    cov_efficacy = 1*(input['cv_efficacy'] == 'Very poorly') + 2*(input['cv_efficacy'] == 'Somewhat poorly') + 3*(input['cv_efficacy'] == 'Somewhat well')+ 4*(input['cv_efficacy'] == 'Very well')
    cov_efficacy_flag = 1*(cov_efficacy == 0)
    
    # Strata for pre-test response
    pre_false = (1*(input['dv_timeline_pre1'] == 'Yes') + 1*(input['dv_send_pre1'] == 'Yes') +
                 1*(input['dv_timeline_pre2'] == 'Yes') + 1*(input['dv_send_pre2'] == 'Yes') )
    pre_true =  (1*(input['dv_timeline_pre3'] == 'Yes') + 1*(input['dv_send_pre3'] == 'Yes') +
                 1*(input['dv_timeline_pre4'] == 'Yes') + 1*(input['dv_send_pre4'] == 'Yes') )

    strat_false0 = 1*(pre_false == 0)
    strat_false1 = 1*(pre_false == 1)
    strat_false2 = 1*(pre_false == 2)
    strat_false3 = 1*(pre_false == 3)
    strat_false4 = 1*(pre_false == 4)
    strat_true0 = 1*(pre_true == 0)
    strat_true1 = 1*(pre_true == 1)
    strat_true2 = 1*(pre_true == 2)
    strat_true3 = 1*(pre_true == 3)
    strat_true4 = 1*(pre_true == 4)

    # Indicators for individual stimuli
    # False stimuli from respective countries
    stimf1 = (1*(input['dv_stimulus_pre1'] == 'FN1') + 1*(input['dv_stimulus_pre2'] == 'FN1'))
    stimf2 = (1*(input['dv_stimulus_pre1'] == 'FN2') + 1*(input['dv_stimulus_pre2'] == 'FN2') +
              1*(input['dv_stimulus_pre1'] == 'FK2') + 1*(input['dv_stimulus_pre2'] == 'FK2'))
    stimf3 = (1*(input['dv_stimulus_pre1'] == 'FN3') + 1*(input['dv_stimulus_pre2'] == 'FN3') +
              1*(input['dv_stimulus_pre1'] == 'FK3') + 1*(input['dv_stimulus_pre2'] == 'FK3'))
    stimf4 = (1*(input['dv_stimulus_pre1'] == 'FN4') + 1*(input['dv_stimulus_pre2'] == 'FN4') +
              1*(input['dv_stimulus_pre1'] == 'FK4') + 1*(input['dv_stimulus_pre2'] == 'FK4'))
    stimf5 = (1*(input['dv_stimulus_pre1'] == 'FN5') + 1*(input['dv_stimulus_pre2'] == 'FN5') +
              1*(input['dv_stimulus_pre1'] == 'FK5') + 1*(input['dv_stimulus_pre2'] == 'FK5'))
    stimf6 = (1*(input['dv_stimulus_pre1'] == 'FN6') + 1*(input['dv_stimulus_pre2'] == 'FN6') +
              1*(input['dv_stimulus_pre1'] == 'FK6') + 1*(input['dv_stimulus_pre2'] == 'FK6'))
    stimf7 = (1*(input['dv_stimulus_pre1'] == 'FN7') + 1*(input['dv_stimulus_pre2'] == 'FN7') +
              1*(input['dv_stimulus_pre1'] == 'FK7') + 1*(input['dv_stimulus_pre2'] == 'FK7'))
    stimf8 = (1*(input['dv_stimulus_pre1'] == 'FN8') + 1*(input['dv_stimulus_pre2'] == 'FN8') +
              1*(input['dv_stimulus_pre1'] == 'FK8') + 1*(input['dv_stimulus_pre2'] == 'FK8'))
    stimf9 = (1*(input['dv_stimulus_pre1'] == 'FN9') + 1*(input['dv_stimulus_pre2'] == 'FN9') +
              1*(input['dv_stimulus_pre1'] == 'FK8') + 1*(input['dv_stimulus_pre2'] == 'FK9'))
    stimf10 = (1*(input['dv_stimulus_pre1'] == 'FN10') + 1*(input['dv_stimulus_pre2'] == 'FN10') +
              1*(input['dv_stimulus_pre1'] == 'FK10') + 1*(input['dv_stimulus_pre2'] == 'FK10'))
    stimf11 = (1*(input['dv_stimulus_pre1'] == 'FN11') + 1*(input['dv_stimulus_pre2'] == 'FN11') +
              1*(input['dv_stimulus_pre1'] == 'FK11') + 1*(input['dv_stimulus_pre2'] == 'FK11'))
    stimf12 = (1*(input['dv_stimulus_pre1'] == 'FN12') + 1*(input['dv_stimulus_pre2'] == 'FN12') +
              1*(input['dv_stimulus_pre1'] == 'FK12') + 1*(input['dv_stimulus_pre2'] == 'FK12'))
    stimf13 = (1*(input['dv_stimulus_pre1'] == 'FN13') + 1*(input['dv_stimulus_pre2'] == 'FN13') +
              1*(input['dv_stimulus_pre1'] == 'FK13') + 1*(input['dv_stimulus_pre2'] == 'FK13'))
    stimf14 = (1*(input['dv_stimulus_pre1'] == 'FN14') + 1*(input['dv_stimulus_pre2'] == 'FN14'))
    stimf15 = (1*(input['dv_stimulus_pre1'] == 'FN15') + 1*(input['dv_stimulus_pre2'] == 'FN15'))
    stimf16 = (1*(input['dv_stimulus_pre1'] == 'FN16') + 1*(input['dv_stimulus_pre2'] == 'FN16'))
    # True stimuli from respective countries
    stimt1 = (1*(input['dv_stimulus_pre3'] == 'TN1') + 1*(input['dv_stimulus_pre4'] == 'TN1') +
              1*(input['dv_stimulus_pre3'] == 'TK1') + 1*(input['dv_stimulus_pre4'] == 'TK1'))
    stimt2 = (1*(input['dv_stimulus_pre3'] == 'TN2') + 1*(input['dv_stimulus_pre4'] == 'TN2') +
              1*(input['dv_stimulus_pre3'] == 'TK2') + 1*(input['dv_stimulus_pre4'] == 'TK2'))
    stimt3 = (1*(input['dv_stimulus_pre3'] == 'TN3') + 1*(input['dv_stimulus_pre4'] == 'TN3') +
              1*(input['dv_stimulus_pre3'] == 'TK3') + 1*(input['dv_stimulus_pre4'] == 'TK3'))
    stimt4 = (1*(input['dv_stimulus_pre3'] == 'TN4') + 1*(input['dv_stimulus_pre4'] == 'TN4') +
              1*(input['dv_stimulus_pre3'] == 'TK4') + 1*(input['dv_stimulus_pre4'] == 'TK4'))
    stimt5 = (1*(input['dv_stimulus_pre3'] == 'TN5') + 1*(input['dv_stimulus_pre4'] == 'TN5') +
              1*(input['dv_stimulus_pre3'] == 'TK5') + 1*(input['dv_stimulus_pre4'] == 'TK5'))
    stimt6 = (1*(input['dv_stimulus_pre3'] == 'TN6') + 1*(input['dv_stimulus_pre4'] == 'TN6') +
              1*(input['dv_stimulus_pre3'] == 'TK6') + 1*(input['dv_stimulus_pre4'] == 'TK6'))
    # True stimuli from both countries
    stimb1 = (1*(input['dv_stimulus_pre3'] == 'TB1') + 1*(input['dv_stimulus_pre4'] == 'TB1'))
    stimb2 = (1*(input['dv_stimulus_pre3'] == 'TB2') + 1*(input['dv_stimulus_pre4'] == 'TB2'))
    stimb3 = (1*(input['dv_stimulus_pre3'] == 'TB3') + 1*(input['dv_stimulus_pre4'] == 'TB3'))
    stimb4 = (1*(input['dv_stimulus_pre3'] == 'TB4') + 1*(input['dv_stimulus_pre4'] == 'TB4'))
    stimb5 = (1*(input['dv_stimulus_pre3'] == 'TB5') + 1*(input['dv_stimulus_pre4'] == 'TB5'))
    
    
    # fix this 
    # xt, a vector of numeric covariates; t indexed observation number
    xt = [
          male, 
          age,
          ed,
          urban,
          rel_none, rel_christian, rel_muslim, rel_traditionalist, rel_other, denom_pentecostal, religiosity,
          god,
          locus,
          science,
          dli,
          crt,
          hhi,
          cash,
          occ_0, occ_1, occ_2, occ_3, occ_4, occ_5, occ_6, occ_7, occ_8, occ_9, occ_10, occ_11, occ_12, occ_13,
          hh,
          pol,
          cov_concern,
          cov_info,
          cov_efficacy,
          pre_false,
          pre_true,
          strat_false0, strat_false1, strat_false2, strat_false3, strat_false4,
          strat_true0, strat_true1, strat_true2, strat_true3, strat_true4,
          stimf1, stimf2, stimf3, stimf4, stimf5, stimf6, stimf7, stimf8, stimf9, stimf10, stimf11, stimf12, stimf13, stimf14, stimf15, stimf16,
          stimt1, stimt2, stimt3, stimt4, stimt5, stimt6,
          stimb1, stimb2, stimb3, stimb4, stimb5  
          ]
    

   
    t = 3400  # TODO: This is the index value. Augment this with every observation starting with 0.
    
    # TODO: Read in ACTUAL model object(s).
    #    This is a placeholder, uninformative model.
    #    We can save it as the initial model object on the server, for backup in case there is an overlap between the
    #    first model update and when the next subjects roll in.
    model = (np.zeros((K, p + 1)), np.repeat(np.identity(p + 1)[np.newaxis, :, :], 40, axis=0))
    bandit_model = fit_policytree(np.random.normal(scale=1, size=(40, p)),
                                  np.random.normal(scale=1, size=(40, 40)))
    
    # ASSIGN TREATMENT
    if t < num_init_draws:
        wt = t % K
        pt = [1 / K] * K
    elif t < 3200:
        wt, pt = draw_weighted_ridge_thompson(xt, model, config)
        control = np.random.binomial(1, p=1 / 5)
        if control:
            wt = ctr
        pt[ctr] = 1 / 5 + pt[ctr] * 4 / 5
        pt[nctr] = pt[nctr] * 4 / 5
    else:  # Assign (nearly) deterministically according to policy tree object
        wt = predict_policytree(bandit_model, xt, num_threads=num_threads)[0]
        pt = np.repeat((1 / 10) / 40, K)
        pt[wt] = (1 / 10) / 40 + 9 / 10
        epsilon = np.random.binomial(1, p=1 / 10)
        if epsilon:
            wt = np.random.randint(40)
    
    # Send `wt` as a treatment assignment back to chatfuel as an attribute

    user_id = str(input['messenger user id'])    

    insertRow = [user_id, wt]

    print(insertRow)
    sheet.append_row(insertRow)
    full_dataset.append_row(xt)

if responded == 1 :
    # UPDATE MODEL
    # TODO: only complete this if we have response attributes; check for input['dv_send_post8']
    
    # Response function:
    post_false = (1 * (input['dv_timeline_post5'] == 'Yes') + 1 * (input['dv_send_post5'] == 'Yes') +
                  1 * (input['dv_timeline_post6'] == 'Yes') + 1 * (input['dv_send_post6'] == 'Yes'))
    post_true = (1 * (input['dv_timeline_post7'] == 'Yes') + 1 * (input['dv_send_post7'] == 'Yes') +
                 1 * (input['dv_timeline_post8'] == 'Yes') + 1 * (input['dv_send_post8'] == 'Yes'))
    
    yt = - post_false + 0.5 * post_true
    # TODO Read in all _ORDERED_ historical observations; the below are just randomly generated
    # xs, ys, ws, ps: historical observations
    xs = np.random.normal(scale=1, size=(2000, p))  # history of all covariates up to time t
    ys = np.random.normal(scale=1, size=2000)  # history of all responses up to time t
    ws = np.resize(range(40), 2000)  # history of all treatments up to time t
    ps = np.full((2000, K), 1 / K)  # history of all treatment assignment probabilities up to time t
    
    # Vectors of historical + CURRENT observation
    xs_t = np.vstack((xs, xt))
    ys_t = np.concatenate((ys, [yt]))
    ws_t = np.concatenate((ws, [wt]))
    ps_t = np.vstack((ps, pt))
    balwts = 1 / collect(ps_t, ws_t)
    
    if t in update_times[:-1]:
        lambda_min = fit_ridge_lambda(xs_t, ys_t)
    
        model = update_weighted_ridge_thompson(xs_t, ys_t, ws_t, balwts, lambda_min, K,
                                               intercept=True)
        # TODO: Save updated model object
    
    # Estimate muhat on first split
    if t == tlast:
        # TODO: to save time, we could calculate muhats for each batch and save, so that by the last batch, we are only
        #  learning the *last* batch worth of muhats. Right now, this takes me ~3:30 mins for 2k observations, and this is
        #  by far the most time-consuming aspect of this calculation.
        muhat = forest_muhat_lfo(xs_t, ws_t, ys_t, K, update_times + 1, num_threads=1)
        aipw_scores = aw_scores(ys_t, ws_t, balwts, K=K, muhat=muhat)
        bandit_model = fit_policytree(xs_t, aipw_scores)
        # TODO: Save bandit_model object; pkl?
    
    
    # SAVE DATA?
    # TODO: eventually we should save all ordered data, potentially matched to chatfuel ID, and the policy tree R object
    data = dict(yobs=ys_t, ws=ws_t, xs=xs_t, probs=ps_t, final_bandit_model=bandit_model)