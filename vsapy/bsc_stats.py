import math
from decimal import Decimal
import numpy as np
from scipy import special as scm
import gmpy2
from .vsa_stats import *


def get_perror(nobits, target_hd, alternate_hd, num_trials, vocab_size):
    return perror(nobits, 1.0 - target_hd, 1.0-alternate_hd, num_trials, vocab_size)


def subvec_mean(num_vecs):
    """
    :param num_vecs: This is the number of vectors to be added via majority_sum
    :return: The normalised hamming distance similarity between an individual vector of the sum and the majority_sum.
             0 = exact match. If the vectors being added are 1000 bits long then a result of 0.25 means that
             of the 1000 bits only 250 bits will be different than those contained in each individual vector.
             Or putting it the other way around 75% of the bits in each individual vector in the sum will match to the
             bits in the resultant majority_sum vector.
    """

    if num_vecs == 1:
        return float(1.0)

    # Majority sum needs an odd number of vectors, an additional random vector is used when the sum contains
    # and even number of vectors, hence if we have an even number here it is equivalent to calculating on the
    # num_vecs + 1
    if num_vecs % 2 > 0:  # If odd
        num_vecs -= 1

    P = 0  # Cumulative permutation sum
    for j in range(num_vecs // 2, num_vecs + 1):
        P = P + gmpy2.comb(num_vecs, j)

    for _ in range(num_vecs):
        P /= 2

    return float(P)


def sum_combis(N, k):
    combis = gmpy2.mpz(0)
    for j in range(k, N + 1):
        combis += gmpy2.comb(N, j)
    return float(combis / 2 ** N)


def get_probability_that_bit_is_set(total_subvecs_in_this_sum, num_shared_vecs,
                                    majority_bit_cnt_in_this_shared_vec_instance):
    """

    :param total_subvecs_in_this_sum: Number of sub vecs

    :param num_shared_vecs:  Number of vecs common to some other vector

    :param majority_bit_cnt_in_this_shared_vec_instance:
        Imagine that there are two shared vectors, then this parameter indicates the number of bits that are in the
        majority for this instance of the shared vecs, for example, when num_shared_bits = 2, all possibilities are
            00
            01
            10
            11

        When the '01' instance is being processed this param = 1, when 00 or 11 is being processed this param = 2

    :return: Probability that the bit in Z/Z' will have the same sense as the majority in the shared vecs.
    """
    wining_line = int(total_subvecs_in_this_sum / 2) + 1  # winning count for Majority_Sum
    other_vecs = total_subvecs_in_this_sum - num_shared_vecs

    if majority_bit_cnt_in_this_shared_vec_instance >= wining_line:
        # If shared vecs have put us over the winning line then other_bits='Don't Care'
        return 1.0
    else:
        if float(majority_bit_cnt_in_this_shared_vec_instance) == float(num_shared_vecs) / 2.0:
            # Then all winnig bits come Non-shared-bit-columns
            # If there is a tie in the shared_bits then winner comes solely from other side
            # i.e. the winner is the majority occurrence chosen from (other_vecs ONLY)
            p = 0.5  # when choosing a majority of 1, e.g 3c2 or 11c6 the probability is always p=0.5
        else:
            extra_bits_needed = wining_line - majority_bit_cnt_in_this_shared_vec_instance
            p = sum_combis(other_vecs, extra_bits_needed)

    return p  # Probability that the bit in Z/Z' will have the same sense as the majority in the shared vecs.


def get_one_sided_shared_vec_probability_profile(total_subvecs_in_this_sum, num_shared_vecs):
    """

    This has O(N) time complexity.

    :param total_subvecs_in_this_sum: number of sub_vecs in the sum (i.e. the target_vector=Z)
    :param num_shared_vecs: number of sub_vecs that might be shared in some other Z'
    :return: theoretical hamming similarity
    """

    """--------------------------------------------------------------------------------------------------------------
    Map the number of dominant bits (e.g. 1s) for all possible occurences of num_shared_vecs against frequency of 
    occurence, for example, when num_shared_bits = 2, all possibilities are
        00
        01
        10
        11
                                            10  
                                 00         01         11
    Thus shared_vec_combi_map = [(2, 0.25), (1, 0.5),  (2, 0.25)]
                                     p=25%      p=50%      p=25%

    """
    DEBUG_PREPROC = False
    shared_vec_combi_map = []
    shared_vec_combi_map_check = []
    for i in range((num_shared_vecs+1) // 2, num_shared_vecs+1):
        if DEBUG_PREPROC:
            shared_vec_combi_map_check.append((i, scm.comb(num_shared_vecs, i)))

        shared_vec_combi_map.append((i, scm.comb(num_shared_vecs, i)/2**num_shared_vecs))
    """--------------------------------------------------------------------------------------------------------------"""

    prob_map = []
    for shared_ones_cnt, prob_of_this_shared_bit_instance_occurring in shared_vec_combi_map:
        n_pwr = num_shared_vecs // shared_ones_cnt - 1
        divisor = 2 ** n_pwr
        prob_map.append((get_probability_that_bit_is_set(total_subvecs_in_this_sum, num_shared_vecs, shared_ones_cnt),
                         2.0/float(divisor) * prob_of_this_shared_bit_instance_occurring, shared_ones_cnt))

    """--------------------------------------------------------------------------------------------------------------
    Continuing the example, get_probability_that_bit_is_set() calculates the probability that the bit in Z/Z' 
    will have the same sense as the majority in the shared vecs, for num_shared_bits = 2,

                                 10  
                        00       01          11
    Thus prob_map = [(1, 0.25), (0.5, 0.5),  (1, 0.25)]
            occurs:       25%          50%         25%
       contributes: 100%         50%        100%        example, depends on total_sub_vecs.. and num_shared_vecs

    """
    return prob_map


def get_shared_hamming_similarity(total_no_subvecs_in_V1_sum, num_shared_vecs, total_no_subvecs_in_V2_sum):
    """

    This has O(N) time complexity and is much quicker

    :param total_no_subvecs_in_V1_sum:
    :param num_shared_vecs:
    :param total_no_subvecs_in_V2_sum:
    :return: probability that Z=Z' which is also the hamming similarity
    """

    if num_shared_vecs == 0:
        return 0.5  # If there are no shared vecs then by definition the vectors are orthogonal, i.e. HDSim = 0.5

    assert num_shared_vecs <= min(total_no_subvecs_in_V1_sum, total_no_subvecs_in_V2_sum), \
        "parameter 2, num_shared_vecs can not be greater than total vecs contained in a sum vector."
    v1_probability_map = get_one_sided_shared_vec_probability_profile(total_no_subvecs_in_V1_sum, num_shared_vecs)
    v2_probability_map = get_one_sided_shared_vec_probability_profile(total_no_subvecs_in_V2_sum, num_shared_vecs)

    sum_prob1 = 0
    for i in range(len(v1_probability_map)):
        # The probability of each shared_bit_instance, that is this particular instance combination of 1s and 0's,
        # should always be equal since it depends solely on the number of vecs common to both vectors
        assert v1_probability_map[i][1] == v2_probability_map[i][1], "Shared_probabilities_do_not_match"

        pv11 = v1_probability_map[i][0]  # probability that the bit is set, i.e. Z = 1, in vector 1
        pv22 = v2_probability_map[i][0]  # probability that the bit is set, i.e. Z'= 1, in vector 2


        # The probability that Z=Z' works both ways, that is
        # We need (prob_V1_occurring) * (prob_V2_occurring) + (prob_V1_NOT_occurring) * (prob_V2_NOT_occurring)
        prob1 = pv11 * pv22 + (1 - pv11) * (1 - pv22)

        sum_prob1 += v1_probability_map[i][1] * prob1  # prob of this shared_bit_config * prob of Z=Z'

    return sum_prob1
