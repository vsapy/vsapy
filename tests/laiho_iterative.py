import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', None)
pd.set_option("display.max_rows", None, "display.max_columns", None)

import vsapy as vsa
from vsapy.vsatype import VsaType, VsaBase
from vsapy.bag import *
import numpy as np
from scipy import stats
import timeit
from vsapy.laiho import *
from vsapy.laihox import *


class BinSearch(object):
    def __init__(self, start, end, step_size, threshold, last_win,  show_win=False):
        self.threshold = threshold
        self.low = start
        self.high = end
        self.step_size = step_size
        self.starthigh = end
        self.startlow = start
        self.prev_high = end + 8
        self.mid = int((end-start)/2)+self.low
        self.direction = 0
        self.last_win_value = -1
        self.best_win = -1
        self.lastfail = 10000
        self.last_win_value = last_win
        self.show_win = show_win

    def step_or_quit(self, testval, err):
        if err <= self.threshold:
            # This is a positive result
            # We need to try stepping higher
            # Halving the high-low interval has left us in the same spot therefore we are near the end.
            if self.lastfail == testval + 1:
                # This is the halting condition because we have already had a fail 1 step above this win.
                if self.show_win:
                    print(f"\t\t\t\t Start:{self.startlow}, end={self.starthigh}, start_val={self.mid} :: Win={testval}")
                return testval, True

            # From here on in we need to generate a higher number to asee if we can get a win at a higher value
            self.last_win_value = max(self.last_win_value, testval)
            self.low = testval
            new_testval = int((self.high-self.low)/2) + self.low
            if new_testval <= testval:
                print("\t\t\t\t\t\t Moving high")
                self.high += self.step_size
                new_testval = int((self.high-self.low)/2) + self.low
                return new_testval, False
            else:
                return new_testval, False
        else:
            # this was a negative result
            # we need to try stepping lower
            if self.last_win_value == testval - 1:
                # This is the halting condition because we have already had a win 1 step below this fail.
                if self.show_win:
                    print(f"\t\t\t\t Start:{self.startlow}, end={self.starthigh}, start_val={self.mid} :: Win={self.last_win_value}")
                return self.last_win_value, True

            self.lastfail = min(testval, self.lastfail)
            # From here on in we need to generate a lower value to see if we can win at the lower number
            self.prev_high = testval
            self.high = testval # set high to fail point
            new_testval = int((self.high-self.low)/2) + self.low
            if new_testval >= testval:
                # we must move the high up and try again
                print("\t\t\t\t\t\t Moving Low")
                self.low -= self.step_size
                new_testval = int((self.high-self.low)/2) + self.low
                return new_testval, False
            else:
                return new_testval, False


def check_vecs(sumvec, num_vecs_in_sum, vl, stdev_count, show_hd=False):

    M = len(sumvec)

    exprv = 1/sumvec.bits_per_slot  # Probability of a match between random vectors
    var_rv = M * (exprv * (1-exprv))  # Varience (un-normalised)
    std_rv = math.sqrt(var_rv)        # Stdev (un-normalised)
    hdrv = M/sumvec.bits_per_slot + stdev_count * std_rv  # Un-normalised hsim of two randomvectors adjusted by 'n' stdevs
    errcnt = 0
    for i in range(num_vecs_in_sum):
        hdi = M * vsa.hsim(sumvec, vl[i])  # vl is the list of sub-vectors, sumvec is the iteratively built bundle vec
        if hdi <= hdrv:  # If hdi <= a random vector we have failed to decode.
            errcnt += 1
            if show_hd:
                print(f"\t\t\thd{i}={hdi}, rv={hdrv}")
            break
        else:
            if show_hd:
                print(f"hd{i}={hdi}, rv={hdrv}")

    if errcnt > 0.0:
        if show_hd:
            print(f'=== ERRORS({errcnt}) ================\n')
        return sumvec, errcnt
    elif show_hd:
        print('========================\n')

    return sumvec, 0.0


def create_iterative_sum(num_vecs, vl):
    sumx = vl[0][:]
    for j in range(1, num_vecs):
        sumx = vl[0].sum([sumx, vl[j]])

    return sumx


def iter_trails(slots, bits_per_slot, trails, start, end, stdev_cnt, vsa_type):
    assert vsa_type == VsaType.Laiho or vsa_type == VsaType.LaihoX, "vsa_type must be Laiho or LaihoX."
    errs = 0
    binsrch = BinSearch(start, end, 10, 0, -1, True)
    iter_sum = binsrch.mid
    passfail = False
    while not passfail:
        print(f"\niter_test={iter_sum}")
        for k in range(trails):
            vlist = vsa.randvec((iter_sum, slots), vsa_type=vsa_type, bits_per_slot=bits_per_slot)
            sumv = create_iterative_sum(iter_sum, vlist)
            sumv, errs = check_vecs(sumv, iter_sum, vlist, stdev_cnt)
            if errs > 0:
                #rint(f'\nNo iterations: {k}  === ERRORS({errs}) ================\n')
                break
            else:
                if k != 0 and k % 20 == 0:
                    print("")
                print(f'{k}.', end='')
        iter_sum, passfail = binsrch.step_or_quit(iter_sum, errs)

    return iter_sum


def plot_results(data, xseries_name, chart_title, fname, save_csv=True, path="./", logX=False, logY=False):
    # Ensure output directory exists
    if path != "./":
        Path(path).mkdir(parents=True, exist_ok=True)

    rf = pd.DataFrame(data)
    timestr1 = time.strftime("%Y%m%d-%H%M%S").strip()
    fname = f"{timestr1}_" + fname  # Add a timestamp to the front of the file name
    if save_csv:
        csv_header = True

        with open(path+fname+".csv", 'w') as f:
            rf.to_csv(f, header=csv_header)
        csv_header = False

    print(rf)
    ax = rf.plot(x=xseries_name, marker='.', logx=logX, logy=logY)
    ax.grid('on', which='major', axis='x')
    ax.grid('on', which='major', axis='y')
    ax.set_title(chart_title)

    if logX or logY:
        plt.savefig(path+fname+"_log.png")
    else:
        plt.savefig(path+fname+".png")
    plt.show()

if "__main__" in __name__:

    bsc_dim = 10000
    bsc_dim_str = f"{int(bsc_dim/1000)}K"
    stdevs = 4.4
    pretrials = 200
    trails = 5000
    xvals = [1000]
    vals = [200]
    valsX = [200]
    first_pass = True
    for B in [4096, 2048, 1024, 512, 512-64, 256+64, 256, 128, 64, 32, 16, 8, 4, 2]:
        xvals.append(B)
        M = Laiho.slots_from_bsc_vec(bsc_dim, B)
        print(f"Bits_per_Slot = {B}")
        # first get to the correct ballpark using only 1 trial
        if first_pass:
            start = 40
            end = 200
            startX = 40
            endX = 200
        else:
            start = max(vals[-1]-5, 0)
            end = vals[-1]
            startX = max(valsX[-1]-5, 0)
            endX = valsX[-1]
        capacity = iter_trails(M, B, pretrials, start, end, stdevs, VsaType.Laiho)
        capacityX = iter_trails(M, B, pretrials, startX, endX, stdevs, VsaType.LaihoX)
        if pretrials >= trails:
            vals.append(capacity)
            valsX.append(capacityX)
        else:
            start = max(capacity-20, 0)
            end = capacity
            startX = max(capacityX-20, 0)
            endX = capacityX

            print(f"Laiho starting main search. Start/end=({start}, {end}).")
            vals.append(iter_trails(M, B, trails, start, end, stdevs, VsaType.Laiho))
            print(f"LaihoX starting main search. Start/end=({startX}, {endX}).")
            valsX.append(iter_trails(M, B, trails, startX, end, stdevs, VsaType.LaihoX))

    data = {"Bits_per_slot": xvals[1:], "Laiho capacity": vals[1:], "LaihoX capacity": valsX[1:]}


    fname = f"Laiho_iter_T{trails}_Stdev({stdevs})"
    outputh_path = "./data/laiho_iter_test/"
    chart_title = f"Laiho/X iterative capacity. BSC equiv={bsc_dim_str}, \nTrails={trails}, Perr={stdevs}stdevs."
    plot_results(data, "Bits_per_slot", chart_title, fname, True, path=outputh_path, logX=False)
    plot_results(data, "Bits_per_slot", chart_title, fname, False, path=outputh_path, logX=True)
    quit()