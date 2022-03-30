from .tern import *

class TernZero(Tern):
    vsatype = VsaType.TernZero


    @classmethod
    def unpackbits(cls, v):
        """
        Unpacks the binary vector packed via TernZero.packbits() as a TernZero vector.
        :param v: ndarray, packed vector is concatenation of binary vector and zerobits vector.
        :type v: ndarray type uint8
        :return: TernZero vecor with zeros set correctly.
        :rtype: TernZero vector.
        """
        vbin = np.unpackbits(v).astype('int8')
        # split off the zeros
        zero_bits = vbin[len(vbin)//2:]
        v1 = vbin[:len(vbin)//2]
        v1[v1 == 0] = -1 # All zeros represent -1 or zero. set all to -1.
        v2 = np.sum([v1, zero_bits], axis=0)  # There are 1's in ZeroBits where output vector should be zero
        return VsaBase(v2, vsa_type=VsaType.TernZero)

    @classmethod
    def packbits(cls, v):
        """
        To keep track of the zeros we extend the vector with a mask that can be used to recover them.
        This makes the packed vector twice as long as an ordinary Tern or BSC vec.
        :param v: TernZero vec to be packed
        :type v: TernZero
        :return: packed vector.
        :rtype: numpy unit8
        """
        # Get a mask of all of the zeros
        zero_bits = (v == 0)
        v[v == -1] = 0  # convert -1 to zeros
        # concat the zero mask to the end of v
        outv = np.concatenate((v, zero_bits))
        return np.packbits(outv)

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
