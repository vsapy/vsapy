
import vsapy as vsa
from vsapy import *
#from vsapy.vsatype import VsaType, VsaBase
from vsapy.vsatype import *
from vsapy.laiho import Laiho
import numpy as np


print("\n============ Test creating a VsaType from numpy generator =====================")
vd = 32  # vector dimension
bits_per_slot = 4  # Laiho bits per slot.
slots = Laiho.slots_from_bsc_vec(vd, bits_per_slot)
print("Creating vec of each type, lenth={}, filled with zeros".format(vd))
for vt in VsaType:
    if Laiho.is_laiho_type(vt):
        v = VsaBase(np.zeros((slots,bits_per_slot)), vsa_type=vt, bits_per_slot=bits_per_slot)
    else:
        v = VsaBase(np.zeros(vd), vsa_type=vt)
    print("\t{}: Len={}, dtype={}, {}".format(vt.name, len(v), v.dtype, v))


# Test VSA Type conversions
print("\n============ Test VsaType Conversions =====================")
trunc_view = 10
vd = 1000  # vector dimension
bits_per_slot = 16  # Laiho bits per slot.
slots = Laiho.slots_from_bsc_vec(vd, bits_per_slot)
test_types = [VsaType.BSC, VsaType.Tern, VsaType.TernZero, VsaType.HRR]
for vt in VsaType:
    print()
    if Laiho.is_laiho_type(vt):
        a = vsa.randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
        b = vsa.randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
    else:
        a = vsa.randvec(vd, vsa_type=vt)
        b = vsa.randvec(vd, vsa_type=vt)
    print(f"{str(VsaType(vt))}")
    #v1 = vsa.normalize(a+b, 2)
    v1 = vsa.BagVec([a, b]).myvec
    for vtt in test_types:
        try:
            t1 = vsa.to_vsa_type(v1, vsa_type=vtt)
            print("Convert {} to {} = \n\t\t{}\n\t\t{}".format(v1.vsatype.name, t1.vsatype.name, v1[:trunc_view], t1[:trunc_view]))
        except:
            print("\nConvert {} to {}: FAILED".format(str(VsaType(vt)), str(VsaType(vtt))))


# Test binding
print("\n============ Test BIND / UNBIND =====================")
vd=2048  # vector dimension
bits_per_slot = 32  # Laiho bits per slot.
slots = Laiho.slots_from_bsc_vec(vd, bits_per_slot)
for vt in VsaType:
    if Laiho.is_laiho_type(vt):
        v1 = vsa.randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
        v2 = vsa.randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
    else:
        v1 = vsa.randvec(vd, vsa_type=vt)
        v2 = vsa.randvec(vd, vsa_type=vt)
    print("\n{}: compare self = {:0.4f}".format(vt.name, vsa.hsim(v1, v1)))
    v3 = vsa.bind(v1, v2)
    print("{}: bind test v3 = v1 * v2".format(vt.name))
    print("\t{}: bind test v1 not similar to v3, hdsim = {:0.4f}".format(vt.name, vsa.hsim(v3, v1)))
    print("\t{}: bind test v2 not similar to v3, hdsim = {:0.4f}".format(vt.name, vsa.hsim(v3, v2)))
    print("\t{}: unbind test v3*V2->v1 = {:0.4f}".format(vt.name, vsa.hsim(v1, vsa.unbind(v3, v2))))
    print("\t{}: unbind test v3*V1->v2 = {:0.4f}".format(vt.name, vsa.hsim(v2, vsa.unbind(v3, v1))))


# Test binding
print("\n===================== Test Simple Concept / UNBIND =====================")
vd=10000 # vector dimension
bits_per_slot = 256  # Laiho bits per slot.
slots = Laiho.slots_from_bsc_vec(vd, bits_per_slot)
num_vecs = 10
for vt in VsaType:
    if Laiho.is_laiho_type(vt):
        vecs = vsa.randvec((num_vecs, slots), vsa_type=vt, bits_per_slot=bits_per_slot)
    else:
        vecs = vsa.randvec((num_vecs, vd), vsa_type=vt)
    bag = vsa.BagVec(vecs)
    bag_vec = bag.myvec
    i = 0
    if vt == VsaType.HRR:
        # Here we see calc expected_hd of an orthogonal vector because we don't know how to calc expected_hd
        expected_hd = "> {:0.4f} orthogonal vector".format(vsa.hsim(vsa.randvec(vd, vsa_type=VsaType.HRR), bag_vec))
    else:
        if vt == VsaType.LaihoX:
            expected_hd = "<--> {:0.4f}=expected **from Laiho**".format(vsa.subvec_mean(bag))
        else:
            expected_hd = "<--> {:0.4f}=expected".format(vsa.subvec_mean(bag))

    print("{}:Bundling {} sub-vectors into 'bag_vec'".format(vt.name, num_vecs))
    for v in vecs:
        print("\t{}: probe bag test hdsim(v{}, bag_vec)={:0.4f} {}".format(vt.name, i, vsa.hsim(v, bag_vec), expected_hd))
        i += 1
    print()
