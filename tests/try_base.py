import base.vsapy as vsa
from base.vsatype import *

import numpy as np


print("\n============ Test creating a VsaType from numpy generator =====================")
vd = 5  # vector dimension
print("Creating vec of each type, lenth={}, filled with zeros".format(vd))
for vt in VsaType:
    v = VsaBase(np.zeros(vd), vsa_type=vt)
    print("\t{}: Len={}, dtype={}, {}".format(vt.name, len(v), v.dtype, v))


# Test VSA Type conversions
print("\n============ Test VsaType Conversions =====================")
trunc_view = 10
vd = 1000  # vector dimension
for vt in VsaType:
    print()
    v1 = vsa.randvec(vd, vsa_type=vt)
    for vtt in VsaType:
        try:
            t1 = vsa.to_vsa_type(v1, vsa_type=vtt)
            print("Convert {} to {} = \n\t\t{}\n\t\t{}".format(v1.vsatype.name, t1.vsatype.name, v1[:trunc_view], t1[:trunc_view]))
        except:
            print("Convert {} to {}: FAILED".format(vt.name, vtt.name))


# Test binding
print("\n============ Test BIND / UNBIND =====================")
vd=2048  # vector dimension
for vt in VsaType:
    v1 = vsa.randvec(vd, vsa_type=vt)
    print("\n{}: compare self = {:0.4f}".format(vt.name, vsa.hsim(v1, v1)))
    v2 = vsa.randvec(vd, vsa_type=vt)
    v3 = vsa.bind(v1, v2)
    print("{}: bind test v3 = v1 * v2".format(vt.name))
    print("\t{}: bind test v1 not similar to v3, hdsim = {:0.4f}".format(vt.name, vsa.hsim(v1, v3)))
    print("\t{}: bind test v2 not similar to v3, hdsim = {:0.4f}".format(vt.name, vsa.hsim(v1, v3)))
    print("\t{}: unbind test V2*v3->v1 = {:0.4f}".format(vt.name, vsa.hsim(v1, vsa.unbind(v2, v3))))
    print("\t{}: unbind test V1*v2->v2 = {:0.4f}".format(vt.name, vsa.hsim(v1, vsa.unbind(v2, v3))))


# Test binding
print("\n===================== Test Simple Concept / UNBIND =====================")
vd=2048 # vector dimension
num_vecs = 10
for vt in VsaType:
    vecs = vsa.randvec((num_vecs, vd), vsa_type=vt)
    bag_vec = np.sum(vecs, axis=0)
    bag_vec = vsa.normalize(bag_vec, seqlength=len(vecs))
    i = 0
    if vt != VsaType.HRR:
        expected_hd = "<--> {:0.4f}=expected".format(1.0 - vsa.get_hd_threshold(num_vecs))
    else:
        # Here we see calc expected_hd of an orthogonal vector because we don't know how to calc expected_hd
        expected_hd = "> {:0.4f} orthogonal vector".format(vsa.hsim(vsa.randvec(vd, vsa_type=VsaType.HRR), bag_vec))

    print("{}:Binding {} sub-vectors into 'bag_vec'".format(vt.name, num_vecs))
    for v in vecs:
        print("\t{}: probe bag test hdsim(v{}, bag_vec)={:0.4f} {}".format(vt.name, i, vsa.hsim(v, bag_vec), expected_hd))
        i += 1
    print()
