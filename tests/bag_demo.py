import numpy as np
import vsapy as vsa
from vsapy.vsatype import VsaType, VsaBase
from vsapy.bag import BagVec




if "__main__" in __name__:

    # create some word vectors

    # To do this at the basic level, 'i.e by hand'
    # 1 assign each letter of the alphabet a random vector
    symbols = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.;:,_'!?-[]&*"
    symbol_dict = {}
    for a in symbols:
        symbol_dict[a] = vsa.randvec(10000)

    # 2 Now we can create a bag of lettters
    abcde_veclist = [symbol_dict[c] for c in list("abcde")]

    # 3 Now we sum them to make an un-normalized bag.
    abcde_sumv = vsa.sum(abcde_veclist, axis=0)
    abcde_bag = vsa.normalize(abcde_sumv, len(abcde_veclist))

    # 3 Now we can test for the presence of each symbol
    for c in symbols[1:11]:  # just compare first 10 symbols (skipping space) so we don't get a giant list
        hs = vsa.hsim(symbol_dict[c], abcde_bag)
        print(f"{c}={hs:0.4f} <--{ ' match' if hs > 0.53 else 'no match' }")

    print("notice that the matches, {a,b,c,d,e} all have similar hamming similarities.")


    msg = "\nTo do this with my bag implementation we can reuse the the symbol_dict and abcde_veclist created above."
    print(msg)
    abcde_bag1 = BagVec(abcde_veclist)
    for c in symbols[1:11]:
        # Note that we want to reference the vsa vector of class BagVec which is self.myvec, i.e. abcde_bag1.myvec
        hs = vsa.hsim(symbol_dict[c], abcde_bag1.myvec)
        print(f"{c}={hs:0.4f} <--{ ' match' if hs > 0.53 else 'no match' }")

    print("notice that the matches, {a,b,c,d,e} all have similar hamming similarities.")

    msg = "\nif we make a bag that has repeats then, unless we deal with repeats correctly we will get a different \n" \
          "result. Lets swap out the letter 'd' for an 'a'"
    abcde_veclist[3] = symbol_dict['a']
    print(msg)
    abcde_bag2 = BagVec(abcde_veclist)
    for c in symbols[1:11]:
        # Note that we want to reference the vsa vector of class BagVec which is self.myvec, i.e. abcde_bag1.myvec
        hs = vsa.hsim(symbol_dict[c], abcde_bag2.myvec)
        print(f"{c}={hs:0.4f} <--{ ' match' if hs > 0.53 else 'no match' }")

    print("notice that the 'a' now has a higher hsim than {b,c,e} and that 'd' is no match.")

    msg = "\nIf we make go one step futher we will be introuble. \n" \
    "Let's additionally swap out the letter 'e' for an 'a'. So we know have 'abcaa'. \n" \
    "This will result in only 'a' being detected"
    print(msg)
    abcde_veclist[4] = symbol_dict['a']
    abcde_bag3 = BagVec(abcde_veclist)
    for c in symbols[1:11]:
        # Note that we want to reference the vsa vector of class BagVec which is self.myvec, i.e. abcde_bag1.myvec
        hs = vsa.hsim(symbol_dict[c], abcde_bag3.myvec)
        print(f"{c}={hs:0.4f} <--{ ' match' if hs > 0.53 else 'no match' }")

    print("notice that only 'a' matches! Do you know why?")

    quit()