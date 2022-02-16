import os
from os.path import exists
import pickle


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


def deserialise_object(picklename, default_obj=None):
    # load
    if os.path.isfile(picklename):
        with open(picklename, "rb") as f:
            obj = pickle.load(f)
            f.close()
    else:
        obj = default_obj
    return obj


