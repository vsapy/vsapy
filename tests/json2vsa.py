# This tests shows that we can build comparable service descriptions from a nested JSON service description.
# The inovation for a distributed environment is field_name_to_vec(), which shows how we can build unique role_vectors
# from anywhere without the need for a central repositor or broadcasting role_vectors around the place :-)

import json
import numpy as np
from tests.build_docs import VsaTokenizer
from vsapy.role_vectors import *
from vsapy.cspvec import *


def keyname2vsa(name, vec_alphabet):
    """
    This method will always create a UNIQUE_role_vector given a string that corresponds to a JSON fieldName.
    The beauty about this encoding method is that it will work in a distributed environment, all that needs
    to be agreed is that each distributed service building a vector for itself uses the same vector_alphabet.

    :param name: field-name, its better to treat these as case insensitive since two names with the same spelling
                 ought to mean the same thing

    :param vec_alphabet: a dictionary of unique vectors one for each letter of allowable char in a field name.
                         This is agreed upon by all remote locations, either by using the same seed for the random
                         number generator, or by being downloadable.

    :return: A unique 'atomic' role vector that will not correlate to any other role vector.
    """

    n = name.lower()
    v = vec_alphabet[n[0]]
    shift = 0
    for c in n[1:]:
        # By using ROLL(shift) for each position we enable unique encoding of words such as 'AA' and 'AAA'
        shift += 1
        v = vsa.bind(v, np.roll(vec_alphabet[c], shift))

    return v


def json2vsa(json_input, vsa_tok):
    """

    :param json_input: Nested JSON description of service, fields->unique_role_vectors, values->vectors from words.
    :param vsa_tok:  VsaTokenizer class to help building vectors from words
    :return: a list of tuples (str, vec) - the individual feature_vectors bound to the values roles.
             the first element of each tuple t[0] is a string description of how the vector was built.
             The right most word is the vector_value, the other words are unique role_vectors representing each
             JSON fieldName '*' represents XOR. XOR is done in pairs from right to left, for example;

                     service * service_inputs * input_data_type * char64jpg
             is calculated in the order
                     (service * (service_inputs * (input_data_type * char64jpg)))
    """

    symbol_dict = vsa_tok.symbol_dict
    if isinstance(json_input, dict):
        dd = []
        for k, v in list(json_input.items()):
            rv = json2vsa(v, vsa_tok)
            if isinstance(rv, list):
                dd.extend([("{} * {}".format(k, i[0]),
                            # Concantnate XOR field-names with sub role-filler found in i[1]
                            vsa.bind(keyname2vsa(k, symbol_dict), i[1])) for i in rv])
            else:
                dd.append(("{} * {}".format(k, rv[0]), vsa.bind(keyname2vsa(k, symbol_dict), rv[1])))
        return dd
    elif isinstance(json_input, list):
        dd = []
        for item in json_input:
            rv = json2vsa(item, vsa_tok)
            if isinstance(rv, list):
                dd.extend([json2vsa(i, vsa_tok) for i in rv])
            else:
                dd.append(rv)
        return dd
    else:
        if isinstance(json_input, tuple):
            return json_input
        else:
            # Here is where we return the 'VALUE' encoding (rather than the key and sub-key encodings).
            # chunkSentenceVector() isused for this ATM.
            # Note, this is where we would put the google news semantic vector encoding, etc
            return json_input, vsa_tok.chunkSentenceVector(str(json_input)).myvec


def show_featuremap(veclist_desc):
    for item in veclist_desc:
        print(f"\t{item[0]}")

    print("\n")


def build_service_vec_from_json(play, vsa_tok,  show_feature_map=True):
    """

    :param play: Nested JSON description of service
    :param show_feature_map: For DEBUG, True to print how feature was built.
           Note: vectors are XOR'd in pairs from right to left.

    :return: compound  vector containing subfeatures separated as per field_name_role_vectors
    """
    veclist = json2vsa(play, vsa_tok)
    _, _, service_vec = BagVec.bundle([t[1] for t in veclist], len(veclist))
    if show_feature_map:
        show_featuremap(veclist)
    return service_vec


def read_json_file(fname):
    infile = open(fname, 'r')
    jdict = json.load(infile)
    infile.close()
    return jdict


def build_service_vec_from_file(fname, vsa_tok, show_feature_map=True):
    play = read_json_file(fname)
    return build_service_vec_from_json(play, vsa_tok, show_feature_map)


def main():
    vsa_type = VsaType.BSC
    if vsa_type == VsaType.Laiho or vsa_type == VsaType.LaihoX:
        role_vecs = create_role_data(vec_len=1000, rand_seed=None, force_new_vecs=True,
                                     vsa_type=vsa_type, bits_per_slot=1024)
    else:
        role_vecs = create_role_data(data_files=None, vec_len=10000, rand_seed=123, vsa_type=vsa_type)

    skip_words = {}
    vsa_tok = VsaTokenizer(role_vecs, _usechunksforwords=False,
                           allow_skip_words=False, skip_words=skip_words,
                           skip_word_criterion=lambda w: False)  # In this case, the lambda is just disabling skip_words

    print("\n\n\nBuilding object_detector_1.json")
    objdetect_v1 = build_service_vec_from_file('data/json_samples/object_detector_1.json', vsa_tok)
    print("\nBuilding data/json_samples/object_detector_2.json")
    objdetect_v2 = build_service_vec_from_file('data/json_samples/object_detector_2.json', vsa_tok)
    print("\n")
    print(f"Comparing Objdect_1 vs Objdect_2 = {vsa.hsim(objdetect_v1, objdetect_v2):0.4f}")

    # Compare first two camera
    print("\nBuilding data/json_samples/tfl_camera_02151.json")
    tfl_camera_02151 = build_service_vec_from_file('data/json_samples/tfl_camera_02151.json', vsa_tok)
    print("\nBuilding data/json_samples/tfl_camera_02158.json")
    tfl_camera_02158 = build_service_vec_from_file('data/json_samples/tfl_camera_02158.json', vsa_tok)
    print("\nBuilding data/json_samples/tfl_camera_07450.json")
    tfl_camera_07450 = build_service_vec_from_file('data/json_samples/tfl_camera_07450.json', vsa_tok)
    print("\nBuilding data/json_samples/tfl_camera_08858.json")
    tfl_camera_08858 = build_service_vec_from_file('data/json_samples/tfl_camera_08858.json', vsa_tok)

    if vsa_type == VsaType.LaihoX:
        note = " (Note: LaihoX thresholds use Laiho estimator and are not accurate)."
    else:
        note = ""
    print(f"\nCompare Stuff that SHOULD match, HD > {vsa.random_threshold(objdetect_v1)}{note}")
    print(f"\tComparing Objdect_1 vs Objdect_2 = {vsa.hsim(objdetect_v1, objdetect_v2):0.4f}")
    print(f"\tComparing tfl_camera_02151 vs tfl_camera_02158 = {vsa.hsim(tfl_camera_02151, tfl_camera_02158):0.4f}")
    print(f"\tComparing tfl_camera_02151 vs tfl_camera_07450 = {vsa.hsim(tfl_camera_02151, tfl_camera_07450):0.4f}")
    print(f"\tComparing tfl_camera_02151 vs tfl_camera_08858 = {vsa.hsim(tfl_camera_02151, tfl_camera_08858):0.4f}")

    print(f"\tComparing tfl_camera_02158 vs tfl_camera_07450 = {vsa.hsim(tfl_camera_02158, tfl_camera_07450):0.4f}")
    print(f"\tComparing tfl_camera_02158 vs tfl_camera_08858 = {vsa.hsim(tfl_camera_02158, tfl_camera_08858):0.4f}")

    print(f"\tComparing tfl_camera_07450 vs tfl_camera_08858 = {vsa.hsim(tfl_camera_07450, tfl_camera_08858):0.4f}")

    print("\nCompare WITH Self for best match, HD = 1.00")
    print(f"\tComparing Objdect_1 with self = {vsa.hsim(objdetect_v1, objdetect_v1):0.4f}")
    print(f"\tComparing Objdect_2 with self = {vsa.hsim(objdetect_v2, objdetect_v2):0.4f}")
    print(f"\tComparing tfl_camera_02151 with self = {vsa.hsim(tfl_camera_02151, tfl_camera_02151):0.4f}")
    print(f"\tComparing tfl_camera_02158 with self = {vsa.hsim(tfl_camera_02158, tfl_camera_02158):0.4f}")
    print(f"\tComparing tfl_camera_07450 with self = {vsa.hsim(tfl_camera_07450, tfl_camera_07450):0.4f}")
    print(f"\tComparing tfl_camera_08858 with self = {vsa.hsim(tfl_camera_08858, tfl_camera_08858):0.4f}")

    print(f"\nCompare Stuff that should NOT match, HD < {vsa.random_threshold(objdetect_v1)}{note}")
    print(f"\tComparing objdetect_v1 vs tfl_camera_02151 = {vsa.hsim(objdetect_v1, tfl_camera_02151):0.4f}")
    print(f"\tComparing objdetect_v1 vs tfl_camera_02158 = {vsa.hsim(objdetect_v1, tfl_camera_02158):0.4f}")
    print(f"\tComparing objdetect_v1 vs tfl_camera_07450 = {vsa.hsim(objdetect_v1, tfl_camera_07450):0.4f}")
    print(f"\tComparing objdetect_v1 vs tfl_camera_08858 = {vsa.hsim(objdetect_v1, tfl_camera_08858):0.4f}")

    print(f"\tComparing objdetect_v2 vs tfl_camera_02151 = {vsa.hsim(objdetect_v2, tfl_camera_02151):0.4f}")
    print(f"\tComparing objdetect_v2 vs tfl_camera_02158 = {vsa.hsim(objdetect_v2, tfl_camera_02158):0.4f}")
    print(f"\tComparing objdetect_v2 vs tfl_camera_07450 = {vsa.hsim(objdetect_v2, tfl_camera_07450):0.4f}")
    print(f"\tComparing objdetect_v2 vs tfl_camera_08858 = {vsa.hsim(objdetect_v2, tfl_camera_08858):0.4f}")


if __name__ == "__main__":
    main()
