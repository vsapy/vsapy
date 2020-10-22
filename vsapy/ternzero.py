from .tern import *

class TernZero(Tern):
    vsatype = VsaType.TernZero


    @classmethod
    def normalize(cls, sv, seqlength=None, rv=None):
        """
        Normalize the VSA vector
        :param a: input VSA vector
        :param seqlength: not used
        :param rv: Optional random vector, used for splitting ties on binary and ternary VSA vectors.
        :return: new VSA vector
        """
        v = sv.copy()  # we don't want to change sv

        # Zeros are allowe to remain in the vector.
        v[v < 0] = -1.0  # all elements < 0 become -1
        v[v > 0] = 1.0  # all elements > 0 become +1
        return v

    @classmethod
    def reset_zeros_normalize(cls, sv,  rv=None):
        """
        Reset any zeros in the vector and normalise other element values..
        :param sv: input VSA vector
        :param rv: Optional random vector, used for splitting ties on binary and ternary VSA vectors.
        :return: new VSA vector
        """
        v = sv.copy()  # we don't want to change sv
        if rv is None:
            rv = cls.randvec(len(v), vsa_type=sv.vsa_type)
        v[v == 0] = rv[v == 0]  # Randomise the zeros to +1 or -1
        v[v < 0] = -1.0  # all elements < 0 become -1
        v[v > 0] = 1.0  # all elements > 0 become +1
        return v
