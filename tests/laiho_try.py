import timeit

from vsapy import randvec, hsim, BagVec, bind, unbind, VsaType
from vsapy.laiho import Laiho
from vsapy.laihox import LaihoX


if __name__ == "__main__":
    print("Test performance of Laiho/X bundling...")
    num_vecs = 10000
    num_sum = 80
    trials = 400
    vsa_type = VsaType.LaihoX
    bits_per_slot = 1024
    starttime = timeit.default_timer()
    vlist = randvec((num_vecs, 1000), vsa_type=vsa_type, bits_per_slot=bits_per_slot)
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
    print(f"hs_bag0 = {hsim(bagv.myvec, vlist[0]):0.4f}")
    print(f"hs_bag1 = {hsim(bagv.myvec, vlist[1]):0.4f}")
    print(f"hs_bag_random = {hsim(bagv.myvec, randvec((1,1000), vsa_type=vsa_type, bits_per_slot=bits_per_slot)):0.4f}")


    print("\nTest bind/unbind, bind/unbind is lossless.")
    bound_vec1 = bind(vlist[0], vlist[1])
    bound_vec2 = bind(vlist[1], vlist[0])
    print(f"hs bound1 = bound2 = {hsim(bound_vec1, bound_vec2):0.4f}")

    unbound1 = unbind(bound_vec1, vlist[1])
    print(f"hs unbound1 = {hsim(unbound1, vlist[0]):0.4f}")
    unbound2 = unbind(bound_vec1, vlist[0])
    print(f"hs unbound2 = {hsim(unbound2, vlist[1]):0.4f}")

    print(f"hs unbound1 = {hsim(vlist[0], unbound1):0.4f}")
    print(f"hs unbound2 = {hsim(vlist[1], unbound2):0.4f}")

    quit()

