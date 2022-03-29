import math
from scipy import stats
import vsapy
from .vsatype import *
from .laiho import *


class LaihoX(Laiho):
    vsatype = VsaType.LaihoX


    @classmethod
    def sum1(cls, vlist, *args, **kwargs):
        """
        Maintains vsa_type custom attribute when perfoming numpy.sum()
        Todo: there is probably a better way than this.
        """
        unpacked_list = [v.expandbits(v).flatten() for v in vlist]
        orvec = np.sum(unpacked_list, axis=0).reshape(vlist[0].slots, vlist[0].bits_per_slot)
        outvec = (orvec != 0).argmax(axis=1)

        return VsaBase(np.array(outvec), vsa_type=VsaType.LaihoX, bits_per_slot=vlist[0].bits_per_slot)

    @classmethod
    def sum(cls, vlist, *args, **kwargs):
        """
        Maintains vsa_type custom attribute when perfoming numpy.sum()
        Todo: there is probably a better way than this.
        """
        outvec = np.min(np.array(vlist).T, axis=1)
        return VsaBase(np.array(outvec), vsa_type=VsaType.LaihoX, bits_per_slot=vlist[0].bits_per_slot)

