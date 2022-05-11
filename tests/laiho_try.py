import math
import vsapy as vsa
from vsapy.vsatype import VsaType, VsaBase
from vsapy.bag import *
import numpy as np
from scipy import stats
import timeit
from vsapy.laiho import *
from vsapy.laihox import *

if "__main__" in __name__:

    print("Test performance of Laiho/X bundling...")
    num_vecs = 10000
    num_sum = 80
    trials = 400
    vsa_type = VsaType.LaihoX
    bits_per_slot = 1024
    starttime = timeit.default_timer()
    vlist = vsa.randvec((num_vecs, 1000), vsa_type=vsa_type, bits_per_slot=bits_per_slot)
    print(f"gen_time - Time taken={timeit.default_timer()-starttime}")
    starttime = timeit.default_timer()
    for _ in range(trials):
        sumv = Laiho.sum(vlist[:num_sum])
    new_time = timeit.default_timer()-starttime
    print(f"Laiho_method - Time taken={new_time}")
    starttime = timeit.default_timer()
    for _ in range(trials):
        sumv1 = Laiho.sum1(vlist[:num_sum])
    old_time = timeit.default_timer()-starttime
    print(f"old_method1 - Time taken={old_time}")
    starttime = timeit.default_timer()
    for _ in range(trials):
        sumv2 = LaihoX.sum(vlist[:num_sum])
    old2_time = timeit.default_timer()-starttime
    print(f"LaihoX_method2 - Time taken={old2_time}")

    print(f"speed up LaihoX is {new_time/old2_time:02f}")


    print("\n\nTest detection of bundled vectors.")
    bagv = BagVec(vlist[:30])
    print(f"hs_bag0 = {vsa.hsim(bagv.myvec, vlist[0]):0.4f}")
    print(f"hs_bag1 = {vsa.hsim(bagv.myvec, vlist[1]):0.4f}")
    print(f"hs_bag_random = {vsa.hsim(bagv.myvec, vsa.randvec((1,1000), vsa_type=vsa_type, bits_per_slot=bits_per_slot)):0.4f}")


    print("\nTest bind/unbind, bind/unbind is lossless.")
    bound_vec1 = vsa.bind(vlist[0], vlist[1])
    bound_vec2 = vsa.bind(vlist[1], vlist[0])
    print(f"hs bound1 = bound2 = {vsa.hsim(bound_vec1, bound_vec2):0.4f}")

    unbound1 = vsa.unbind(bound_vec1, vlist[1])
    print(f"hs unbound1 = {vsa.hsim(unbound1, vlist[0]):0.4f}")
    unbound2 = vsa.unbind(bound_vec1, vlist[0])
    print(f"hs unbound2 = {vsa.hsim(unbound2, vlist[1]):0.4f}")

    print(f"hs unbound1 = {vsa.hsim(vlist[0], unbound1):0.4f}")
    print(f"hs unbound2 = {vsa.hsim(vlist[1], unbound2):0.4f}")

    quit()

