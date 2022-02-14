from os.path import exists
import pickle
from vsapy.cspvec import *


def serialise_object(obj, picklename):
    # Create backup
    if exists(picklename):
        backupname = picklename + '.bak'
        if exists(backupname):
            os.remove(backupname)
        os.rename(picklename, backupname)

    # Save
    with open(picklename, "wb") as f:
        pickle.dump(obj, f)

    return


def deserialise_object(picklename, defult):
    # load
    if os.path.isfile(picklename):
        with open(picklename, "rb") as f:
            obj = pickle.load(f)
            f.close()
    else:
        obj = defult
    return obj


def serialise_vec_heirarchy(chunk_heirarchy, pathfn):
    bare_hamlet = BareChunk(chunk_heirarchy)
    serialise_object(bare_hamlet, pathfn)
    return