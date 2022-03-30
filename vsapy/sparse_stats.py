import os
from os.path import exists
import pickle
import time
import timeit
import scipy.misc as sc
import math
import numpy as np
from scipy.stats import binom
import scipy.stats as st
from scipy import special as scm
import scipy.stats as st
from scipy.stats import binom
from scipy.special import factorial, comb, perm
import gmpy2
from .vsa_stats import *
from .helpers import deserialise_object


class IntegerParts(object):
    def __init__(self, elements, counts):
        self.elments = elements
        self.counts = counts

    @property
    def len(self):
        return np.sum(self.counts)


class SnnHd(object):
    def __init__(self, B, m):
        self.B = B
        self.m = m
        self.B_pow_m1 = gmpy2.mpz(B) ** gmpy2.mpz(m - 1)

        self.Pn = {}
        self.AnW = {}
        self.RnW = {}
        self.AnD = {}
        self.RnD = {}

    def calc_prob(self, Pn, C, D):
        hash_C = C.data.tobytes()
        if D == 0:
            W = 1
            C[-1] -= 1  # ** modifying C
            if hash_C in self.AnW:
                An = self.AnW[hash_C]
                Rn = self.RnW[hash_C]
            else:
                An = num_arrangements(C)
                self.AnW[hash_C] = An
                Rn = num_repeats(self.B, C)
                self.RnW[hash_C] = Rn
            C[-1] += 1  # ** fixup C
        else:
            W = C[-1] + 1
            C[-2] -= 1  # ** modifying C
            if hash_C in self.AnD:
                An = self.AnD[hash_C]
                Rn = self.RnD[hash_C]
            else:
                An = num_arrangements(C)
                self.AnD[hash_C] = An
                Rn = num_repeats(self.B, C)
                self.RnD[hash_C] = Rn
            C[-2] += 1  # ** fixup C

        P = Pn * An * Rn / (gmpy2.mpz(W) * self.B_pow_m1)

        return float(P)


def num_patterns(E, C, m):
    F = np.array([gmpy2.factorial(gmpy2.mpz(x)) for x in E])  # factorial(E)
    Fp = np.power(F, C)  #  np.power(F, C)
    Pn = gmpy2.factorial(m-1) / np.product(Fp) # factorial(m - 1) / np.product(Fp)
    return Pn


def num_arrangements(C):
    F = np.array([gmpy2.factorial(gmpy2.mpz(x)) for x in C])  # factorial(C)
    An = gmpy2.factorial(gmpy2.mpz(sum(C))) / np.product(F)
    return An


def num_repeats(B, C):
    Rn = gmpy2.comb(gmpy2.mpz(B) - 1, gmpy2.mpz(sum(C)))
    return Rn


def accel_asc(n, l):
    a = np.zeros(n + 2)
    k = 1
    a[0] = 0
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while x <= y and k < l - 1:
            a[k] = x
            y -= x
            k += 1
        a[k] = x + y
        a[k + 1] = 0     # We want a trailing zero
        yield a[:k + 2]  # We want a trailing zero


def sparse_error(slots, bits, target_hd, vocab_size):
    return perror(slots, target_hd, 1.0/bits, 1e10, vocab_size)


def subvec_mean(m, B, M, use_disk_files=False, diskfile_path='./integer_partitions'):
    P_sum = 0
    if use_disk_files and exists(f"{diskfile_path}/ip{m-1:04d}.bin"):
        return get_snn_hsim_from_file(m, B, M, diskfile_path)

    mysnn_calc = SnnHd(B, m)
    for value in accel_asc(m - 1, B):
        a = value[::-1]
        # need zero on the start of the elements and counts
        elements, counts = np.asarray(np.unique(a, return_counts=True))
        counts[0] = 0

        if True:  # or zpart.len <= B:
            E = elements
            C = counts
            D = 0
            myPn = num_patterns(E, C, m)
            P = mysnn_calc.calc_prob(myPn, C, D)
            P_sum += P
            if P >= 1.0:
                print(m, P, P_sum, 'W')

            # If combi can be a shared win (E[-1] - E[-2] == 1) need to add in this probability also.
            l = len(E) - 1
            if l >= 1 and E[-1] - E[-2] == 1:
                D = 1
                if E[-1] == 1:
                    C[0] = 1  # We are modifying C but it is not used elsewhere after modification.
                P = mysnn_calc.calc_prob(myPn, C, D)
                P_sum += P
                if P >= 1.0:
                    print(m, P, P_sum, 'D')

    return float(P_sum * M)


def get_snn_hsim_from_file(m, B, M, diskfile_path='./integer_partitions'):
    P_sum = 0
    ipart_name = f"{diskfile_path}/ip{m-1:04d}.bin"
    iparts = deserialise_object(ipart_name, None)
    mysnn_calc = SnnHd(B, m)
    for zpart in iparts:
        if zpart.len <= B:
            E = zpart.elments
            C = zpart.counts
            D = 0
            myPn = num_patterns(E, C, m)
            P = mysnn_calc.calc_prob(myPn, C, D)
            P_sum += P
            if P >= 1.0:
                print(m, P, P_sum, 'W')

            # If combi can be a shared win (E[-1] - E[-2] == 1) need to add in this probability also.
            l = len(E) - 1
            if l >= 1 and E[-1] - E[-2] == 1:
                D = 1
                if E[-1] == 1:
                    C[0] = 1  # We are modifying C but it is not used elsewhere after modification.
                P = mysnn_calc.calc_prob(myPn, C, D)
                P_sum += P
                if P >= 1.0:
                    print(m, P, P_sum, 'D')

    return float(P_sum * M)
