import settings
from data_inputs import *
import numpy as np
import itertools as iter


def get_sim_data(param_dict):
    p,w,l = get_input_data(param_dict)

    # forecast error sigma for next 24h
    p_err_sig_1d = param_dict['sig_p']
    w_err_sig_1d = param_dict['sig_w']
    l_err_sig_1d = param_dict['sig_l']
    # forecast error sigma for 25-72h
    p_err_sig_3d = 2*p_err_sig_1d
    w_err_sig_3d = 2*w_err_sig_1d
    l_err_sig_3d = 2*l_err_sig_1d
    # forecast error sigma for 73-168h
    p_err_sig_7d = 3*p_err_sig_1d
    w_err_sig_7d = 3*w_err_sig_1d
    l_err_sig_7d = 3*l_err_sig_1d

    # p_sig = [settings.p_err_sig_1d]*24 + [settings.p_err_sig_3d]*48 + [settings.p_err_sig_7d]*96
    # w_sig = [settings.w_err_sig_1d]*24 + [settings.w_err_sig_3d]*48 + [settings.w_err_sig_7d]*96
    # l_sig = [settings.l_err_sig_1d]*24 + [settings.l_err_sig_3d]*48 + [settings.l_err_sig_7d]*96

    np.random.seed(153)
    nsims = param_dict['sims']
    p_sim = pd.DataFrame({'hour': settings.HOURS, 'solar_fcst': p})
    w_sim = pd.DataFrame({'hour': settings.HOURS, 'wind_fcst': w})
    l_sim = pd.DataFrame({'hour': settings.HOURS, 'elect_fcst': l})
    for sim in range(nsims):
        # p_sim['sim_%d'%sim] = [max(np.random.normal(val, p_sig[idx], 1)[0].round(2),0) for idx,val in enumerate(p)]
        # w_sim['sim_%d'%sim] = [max(np.random.normal(val, w_sig[idx], 1)[0].round(2),0) for idx,val in enumerate(w)]
        # l_sim['sim_%d'%sim] = [np.random.normal(val, l_sig[idx], 1)[0].round(2) for idx,val in enumerate(l)]

        p_err = [np.random.normal(0, p_err_sig_1d, 1)[0]]*24 +\
                [np.random.normal(0, p_err_sig_3d, 1)[0]]*24 +\
                [np.random.normal(0, p_err_sig_3d, 1)[0]]*24 +\
                [np.random.normal(0, p_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, p_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, p_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, p_err_sig_7d, 1)[0]]*24
        p_sim['sim_%d'%sim] = [0 if val == 0 else min(max(val + p_err[idx], 0), param_dict['solar_cap']) for idx,val in enumerate(p)]
        w_err = [np.random.normal(0, w_err_sig_1d, 1)[0]]*24 +\
                [np.random.normal(0, w_err_sig_3d, 1)[0]]*24 +\
                [np.random.normal(0, w_err_sig_3d, 1)[0]]*24 +\
                [np.random.normal(0, w_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, w_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, w_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, w_err_sig_7d, 1)[0]]*24
        w_sim['sim_%d'%sim] = [0 if val == 0 else min(max(val + w_err[idx], 0), param_dict['wind_cap']) for idx,val in enumerate(w)]
        l_err = [np.random.normal(0, l_err_sig_1d, 1)[0]]*24 +\
                [np.random.normal(0, l_err_sig_3d, 1)[0]]*24 +\
                [np.random.normal(0, l_err_sig_3d, 1)[0]]*24 +\
                [np.random.normal(0, l_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, l_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, l_err_sig_7d, 1)[0]]*24 +\
                [np.random.normal(0, l_err_sig_7d, 1)[0]]*24
        l_sim['sim_%d'%sim] = [0 if val == 0 else (val + l_err[idx]) for idx,val in enumerate(l)]

    sim_perm = list(iter.product(range(nsims),repeat=3))

    return p_sim, w_sim, l_sim, sim_perm
