from vsapy.logger_utils import *
from vsapy.helpers import *
from build_docs import buildacts_from_json, xorBind
import vsapy.vsapy as vsa
from vsapy.cspvec import *
from vsapy.bag import BagVec
from build_docs import VsaTokenizer


def search_chunks(top_chunk, target_vec):
    """
    Search a hierarchical chunk tree for best match.
    :param top_chunk:  starting chunk in the tree
    :param target_vec: vector we want to match
    :return: best chunk
    """
    all_chunks = []
    top_chunk.flattenchunkheirachy(all_chunks)

    best_match = None
    max_sim = 0
    for c in all_chunks:
        hs = vsa.hsim(target_vec, c.myvec)
        if hs > max_sim:
            max_sim = hs
            best_match = c

    return best_match, max_sim

if __name__ in "__main__":
    vsa_type = VsaType.LaihoX
    if vsa_type == VsaType.Laiho or vsa_type == VsaType.LaihoX:
        role_vecs = create_role_data(vec_len=1000, rand_seed=None, force_new_vecs=True,
                                     vsa_type=vsa_type, bits_per_slot=1024)
    else:
        role_vecs = create_role_data(data_files=None, vec_len=10000, rand_seed=123, vsa_type=vsa_type)

    vsa_tok = VsaTokenizer(role_vecs, True,
                           allow_skip_words=False, skip_words={},
                           skip_word_criterion=lambda w: False)  # In this case, the lambda is just disabling skip_words

    # Create a sentence encoded with CSPvec encodiing
    sentence = "the cat sat on the mat"
    print(f'\nCreating a sentence: "{sentence}"')
    cat_mat = vsa_tok.chunkSentenceVector("the cat sat on the mat")
    print(f'\nAdding sequence control to the sentence')
    cat_mat_seq = cat_mat.as_seq()  # Add sequence control vectors

    # Now decode the sequence
    print("\nWhen we create the sequence the first step is exposed.")
    print("We can then step thro the sequence...")
    while True:
        best_chunk, hs = search_chunks(cat_mat, cat_mat_seq)
        print(f"{best_chunk.aname}: {hs:0.4f}")
        cat_mat_seq, _ = best_chunk.forward_unbind(cat_mat_seq)
        if best_chunk.check_for_stopvec(cat_mat_seq):
            print("StopVec seen...")
            break

    print("\n\nWe can run the unbind in reverse...")
    cat_mat_seq, _ = best_chunk.reverse_unbind(cat_mat_seq)
    while True:
        best_chunk, hs = search_chunks(cat_mat, cat_mat_seq)
        print(f"{best_chunk.aname}: {hs:0.4f}")
        cat_mat_seq, _ = best_chunk.reverse_unbind(cat_mat_seq)
        if best_chunk.check_for_start_tag_vec(cat_mat_seq):
            print("Start TagVec seen...")
            break

    print("\n\nWe can jump/wind forward a number of steps (less than number of vecs in the vector)...")
    dest_index = 3
    print(f"index position is zero based. Requesting index posn: {dest_index}")
    cat_mat_seq = best_chunk.wind_vec_forward(cat_mat_seq, dest_index)
    best_chunk, hs = search_chunks(cat_mat, cat_mat_seq)
    print(f"{best_chunk.aname}: {hs:0.4f}")

    print("\n\nAnd wind backwards...")
    dest_index = 1
    print(f"index position is zero based. Requesting index posn: {dest_index}")
    cat_mat_seq = best_chunk.wind_vec_forward(cat_mat_seq, dest_index)
    best_chunk, hs = search_chunks(cat_mat, cat_mat_seq)
    print(f"{best_chunk.aname}: {hs:0.4f}")

    quit()
