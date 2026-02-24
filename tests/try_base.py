import numpy as np

from vsapy import BagVec
from vsapy.vsapy import to_vsa_type, subvec_mean, randvec, bind, hsim, unbind
from vsapy.vsatype import VsaType, VsaBase
from vsapy.laiho import Laiho


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
        a = randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
        b = randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
    else:
        a = randvec(vd, vsa_type=vt)
        b = randvec(vd, vsa_type=vt)
    print(f"{str(VsaType(vt))}")
    v1 = BagVec([a, b]).myvec
    for vtt in test_types:
        try:
            t1 = to_vsa_type(v1, new_vsa_type=vtt)
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
        v1 = randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
        v2 = randvec(slots, vsa_type=vt, bits_per_slot=bits_per_slot)
    else:
        v1 = randvec(vd, vsa_type=vt)
        v2 = randvec(vd, vsa_type=vt)
    print("\n{}: compare self = {:0.4f}".format(vt.name, hsim(v1, v1)))
    v3 = bind(v1, v2)
    print("{}: bind test v3 = v1 * v2".format(vt.name))
    print("\t{}: bind test v1 not similar to v3, hdsim = {:0.4f}".format(vt.name, hsim(v3, v1)))
    print("\t{}: bind test v2 not similar to v3, hdsim = {:0.4f}".format(vt.name, hsim(v3, v2)))
    print("\t{}: unbind test v3*V2->v1 = {:0.4f}".format(vt.name, hsim(v1, unbind(v3, v2))))
    print("\t{}: unbind test v3*V1->v2 = {:0.4f}".format(vt.name, hsim(v2, unbind(v3, v1))))


# Test binding
print("\n===================== Test Simple Concept / UNBIND =====================")
vd=10000 # vector dimension
bits_per_slot = 256  # Laiho bits per slot.
slots = Laiho.slots_from_bsc_vec(vd, bits_per_slot)
num_vecs = 10
for vt in VsaType:
    if Laiho.is_laiho_type(vt):
        vecs = randvec((num_vecs, slots), vsa_type=vt, bits_per_slot=bits_per_slot)
    else:
        vecs = randvec((num_vecs, vd), vsa_type=vt)
    bag = BagVec(vecs)
    bag_vec = bag.myvec
    i = 0
    if vt == VsaType.HRR:
        # Here we see calc expected_hd of an orthogonal vector because we don't know how to calc expected_hd
        expected_hd = "> {:0.4f} orthogonal vector".format(hsim(randvec(vd, vsa_type=VsaType.HRR), bag_vec))
    else:
        if vt == VsaType.LaihoX:
            expected_hd = "<--> {:0.4f}=expected **from Laiho**".format(subvec_mean(bag))
        else:
            expected_hd = "<--> {:0.4f}=expected".format(subvec_mean(bag))

    print("{}:Bundling {} sub-vectors into 'bag_vec'".format(vt.name, num_vecs))
    for v in vecs:
        print("\t{}: probe bag test hdsim(v{}, bag_vec)={:0.4f} {}".format(vt.name, i, hsim(v, bag_vec), expected_hd))
        i += 1
    print()
