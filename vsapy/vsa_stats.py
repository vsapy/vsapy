import numpy as np
from scipy.stats import binom


def perror(nobits, target_hd, alternate_hd, num_trials, vocab_size):
    P_hd = binom.pmf(range(0, nobits), nobits, target_hd)
    #P_n = binom.pmf(range(0, nobits), nobits, alternate_hd)
    P_M= (binom.cdf(range(0, nobits), nobits, alternate_hd)) ** vocab_size
    P_match = P_hd*P_M
    #HD0_all = (P_hd * num_trials).astype(int)
    all_wins = (P_match*num_trials).astype(int)
    P_win = np.sum(all_wins)/num_trials
    P_error = 1 - P_win

    return P_error
