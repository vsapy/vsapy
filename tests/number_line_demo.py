import pandas as pd
import matplotlib.pyplot as plt

from vsapy.cspvec import *
from vsapy.vsa_tokenizer import VsaTokenizer
from vsapy.vsapy import NumberLine


def generate_numberline_graph(num_list):
    hd_val = []
    hd_dist = []
    hd_calc = []
    for i in range(len(num_list)):
        print()
        res = []
        for j in range(len(num_list)):
            res.append((vsa.hsim(num_list[i][1], num_list[j][1]), num_list[i][0], num_list[j][0]))

        res.sort(key=lambda x: x[0], reverse=True)
        prev_dist = -1
        for j in range(len(num_list)):
            hd_calc.append(abs(res[j][1] - res[j][2]))
            hd_val.append(res[j][0])
            hd_dist.append(10000 * (1-res[j][0]))
            if res[j][1] < prev_dist:
                print(f"\t{res[j][1]}-{res[j][2]}={res[j][0]:0.4f}")
            else:
                print(f"HSim({res[j][1]},{res[j][2]})={res[j][0]:0.4f}")
            prev_dist = res[j][1]

    data = {"Distance between points on a 1-D numberline": hd_calc, "Hamming Similarirty": hd_val, "Bit Distance": hd_dist}
    df = pd.DataFrame(data)
    print(df)

    dfplt = df.loc[:, ['Distance between points on a 1-D numberline', 'Hamming Similarirty']]
    dfplt.plot.scatter(x='Distance between points on a 1-D numberline', y='Hamming Similarirty',
                       title=f"Hamming Similarity of numberline vectors", subplots=False, marker='.')
    plt.show()


if __name__ in "__main__":
    bsc_dims = 10000
    bits_per_slot = 1024
    standard_devs = 2.05
    vsa_type = VsaType.BSC

    if vsa_type == VsaType.Laiho or vsa_type == VsaType.LaihoX:
        vecdim = Laiho.slots_from_bsc_vec(bsc_dims, bits_per_slot)
        vsa_kwargs = {"vsa_type": vsa_type, "bits_per_slot": bits_per_slot}
    else:
        vecdim = bsc_dims
        vsa_kwargs = {"vsa_type": vsa_type}

    role_vecs = create_role_data(
        data_files={"role_vectors": f"data/role_vectors_{vsa_type}.bin"},
        vec_len=vecdim,
        rand_seed=123,
        **vsa_kwargs)

    vsa_tok = VsaTokenizer(role_vecs, False,
                           allow_skip_words=False, skip_words=False,
                           skip_word_criterion=lambda w: False)  # In this case, the lambda is just disabling skip_words

    MIN_X_POS = 0  # We do not need to start the range at zero
    MAX_X_POS = 500  # We can use a range larger than the vector size provided we specify a 'bits_per_step > 0'
    quantise_steps = 0  # Try a large range and change the quantisation
    plot_step_size = 10
    num_Line = NumberLine(MIN_X_POS, MAX_X_POS, vecdim, vsa_kwargs, quantise_interval=quantise_steps)

    vec_num_list = []
    for i in range(MIN_X_POS, MAX_X_POS+1, plot_step_size if plot_step_size > 0 else 1):
        vec_num_list.append((i, num_Line.number_to_vec(i)))

    generate_numberline_graph(vec_num_list)

    #####################################################################################################
    # Showing the numberline when the range is greater than the number of bits available in our vector.
    #
    MIN_X_POS = 100  # We do not need to start the range at zero
    MAX_X_POS = 30000  # We can use a range larger than the vector size provided we specify a 'bits_per_step > 0'
    quantise_steps = 0  # Try a large range and change the quantisation
    plot_step_size = 300
    num_Line = NumberLine(MIN_X_POS, MAX_X_POS, vecdim, vsa_kwargs, quantise_interval=quantise_steps)

    vec_num_list = []
    for i in range(MIN_X_POS, MAX_X_POS+1, plot_step_size if plot_step_size > 0 else 1):
        vec_num_list.append((i, num_Line.number_to_vec(i)))

    generate_numberline_graph(vec_num_list)
    quit()
