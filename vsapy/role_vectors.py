import threading
import datetime
from enum import IntEnum
import numpy as np
from .vsapy import *
from .vsatype import VsaBase, VsaType
from vsapy.logger_utils import *
log = setuplogs(level='INFO')

from vsapy.bag import *
from vsapy.helpers import *


class RoleVecs(object):
    next_chunk_id = 0  # Used for debug to identify a particular chunk.
    symbols = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.;:,_'!?-[]&*"

    def __init__(self, veclen, random_seed=None, creation_data_time_stamp=None, vsa_type=VsaType.BSC, **kwargs):
        """

        :param veclen: Dimensionality of the vectors to be created.
        :param creation_data_time_stamp:
        """
        if creation_data_time_stamp is None:
            self.creation_data_time_stamp = TimeStamp.get_creation_data_time_stamp()
        else:
            self.creation_data_time_stamp = creation_data_time_stamp

        # self.id_stamp = "NOT SET"
        # if "NOT SET" in self.id_stamp:
        #     self.id_stamp = self.creation_data_time_stamp

        np.random.seed(random_seed)
        self.symbol_dict = createSymbolVectors(RoleVecs.symbols, veclen,
                                               creation_data_time_stamp=creation_data_time_stamp,
                                               vsa_type=vsa_type, **kwargs)
        self.num_dict = create_base_vecs("0", "9", veclen, True,
                                         creation_data_time_stamp=creation_data_time_stamp, vsa_type=vsa_type, **kwargs)

        self.match_message = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # The random alphanumeric match-tag used in workflow requests

        self.id = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # The responder's vector id used in match replies to differentiate between
                                            # responders in a workflow request
                                            # (this is as an alternative to self.role_match_message)

        self.jobid = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # The senders job-id in a workflow request
        self.matchval = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # The hsim match quality in a reply msg
        self.vec_count = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # The number of vecs embedded in this vector
        self.stopvec = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # The chunk stop vector
        self.permVecs = tuple([vsa.randvec(veclen, vsa_type=vsa_type, **kwargs) for _ in range(150)])
        self.parent = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # Used in DAG encoding
        self.child = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # Used in DAG encoding
        # -------------------------------------------------------------------------
        self.tvec_tag = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # Tvec POSITION-Role-vector
        self.current_pindex = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)
        self.pindex_numeric_base = vsa.randvec(veclen, vsa_type=vsa_type, **kwargs)  # Used when representing numbers as a cyclic-shifted vec


# From initialisation we want certain values to agree across all instances of these vector modules
# They are initialised together here in a fixed order.
# Changing the order will stop remote services participating in this scheme from working because
# the vector alphabet and perm vectors / role vectors in class NewChunkPvecs will be different


def checkt_base_objects_are_synced(data_files, vsa_type=VsaType.BSC):
    must_init_flag = False
    data_objects = {}
    first_file_time_stamp = None
    for k, fname in data_files.items():
        obj = deserialise_object(fname)
        if obj is None:
            return True, None

        for k, v in obj.__dict__.items():
            if isinstance(v, VsaBase) and v.vsa_type != vsa_type:
                return True, None

        data_objects[fname[:fname.find('.bin')]] = obj
        try:
            if isinstance(obj, dict):
                this_time_stamp = obj["creation_data_time_stamp"]
            else:
                # obj is a class
                this_time_stamp = obj.creation_data_time_stamp
        except (AttributeError, KeyError) as e:
            log.info(f"Creation date error on object {k}")
            must_init_flag = True
            break

        if first_file_time_stamp is None:
            first_file_time_stamp = this_time_stamp
            continue

        if not TimeStamp.compare_time_stamps(first_file_time_stamp, this_time_stamp):
            log.info(f"Object {k}: time-stamp does not match other objects time-stamps")
            must_init_flag = True
            break

    if not must_init_flag:
        log.info(f"All files appear to be in sync with time-stamp {first_file_time_stamp}")
    else:
        log.info("BASE FILES out of sync, REBUILDING")

    return must_init_flag, data_objects


def create_role_data(data_files=None, vec_len=10000, *args,
                     rand_seed=None, vsa_type=VsaType.BSC, force_new_vecs=False, **kwargs):

    """
    Loads from disk or creates new role vector containers/objects to ensure that all file objects loaded
    were created at the same time.

    If new objects are created serialises each back to disk.

    :param data_files:
    :param vec_len: Dimensionality of vectors to be created
    :param rand_seed: to always create a known set of role vectors based on the order of creation.
    :return: dictionary of created objects
    """

    if data_files is None:
        # role vector files should be added to this list
        data_files = {
            "role_vecs": "role_vectors.bin",
        }

    must_init_flag = True
    if not force_new_vecs:
        must_init_flag, data_objects = checkt_base_objects_are_synced(data_files, vsa_type)
    creation_data_time_stamp = TimeStamp.get_creation_data_time_stamp()

    if must_init_flag:
        # Newly create/recreate each object that contains role vectors that must be in sync with each other.
        role_vecs = RoleVecs(vec_len, random_seed=rand_seed,
                             creation_data_time_stamp=creation_data_time_stamp,
                             vsa_type=vsa_type, **kwargs)
        serialise_object(role_vecs, "role_vectors.bin")
    else:
        role_vecs = data_objects['role_vectors']

    print(f"Loaded role_vectors.bin, time-stamp {role_vecs.creation_data_time_stamp}")

    return role_vecs
