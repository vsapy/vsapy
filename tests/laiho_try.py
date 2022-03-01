import vsapy as vsa
from vsapy.vsatype import VsaType, VsaBase
from vsapy.bag import *
import numpy as np

from vsapy.laiho import *

if "__main__" in __name__:

    num_vecs = 2
    vlist = vsa.randvec((num_vecs,100), vsa_type=VsaType.Laiho, bits_per_vec=20)

    sumv = Laiho.sum(vlist)
    sumvn = Laiho.normalize(sumv)

    bagv = RawVec(vlist)
    # sumv = np.sum(np.array([unpacked0, unpacked1])
    # bundle = RawVec(vlist)


    bound_vec1 = vsa.bind(vlist[0], vlist[1])
    bound_vec2 = vsa.bind(vlist[1], vlist[0])
    print(f"hs = {vsa.hsim(bound_vec1, bound_vec2):0.4f}")


    unbound1 = vsa.unbind(bound_vec1, vlist[1])
    print(f"hs = {vsa.hsim(unbound1, vlist[0]):0.4f}")
    unbound2 = vsa.unbind(bound_vec1, vlist[0])
    print(f"hs = {vsa.hsim(unbound2, vlist[1]):0.4f}")

    quit()

