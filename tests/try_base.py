import base.vsapy as vsa
from base.vsatype import *

import numpy as np


print("\n============ Test creating a VsaType from numpy generator =====================")
vd = 5
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

v1 = vsa.randvec(vd, vsa_type=VsaType.BSC)
v2 = vsa.randvec(vd, vsa_type=VsaType.BSC)

v3 = vsa.normalize(v1 + v2, 2)

print(vsa.hsim(v1, v3))
try:
    print(vsa.hsim(v1, v2))
except:
  print("PASS: Mismatched Types")
try:
    print(vsa.hsim(v1, v2))
except:
  print("PASS: Mismatched Types")

# Test binding
print("\n============ Test BIND / UNBIND =====================")
vd=2048
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


