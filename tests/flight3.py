import json

import numpy as np

from vsapy import Laiho
from vsapy.role_vectors import create_role_data
from vsapy.vsa_tokenizer import VsaTokenizer
from vsapy.vsatype import VsaType
from vsapy.vsapy import hsim

from vsapy.flight_radar_vsa import FlightRadarVsaEncoder, FlightRadarChunkedEncoder

# --- choose representation ---
vsa_type = VsaType.Tern   # or VsaType.BSC

# --- build shared alphabet/role vectors (your existing pattern) ---
if vsa_type in (VsaType.Laiho, VsaType.LaihoX):
    # role_vecs = create_role_data(vec_len=1000, rand_seed=None, force_new_vecs=True,
    #                              vsa_type=vsa_type, bits_per_slot=1024)
    bits_per_slot = 1024
    bsc_dim = 10000
    vec_len = Laiho.slots_from_bsc_vec(bsc_dim, bits_per_slot)

    role_vecs = create_role_data(vec_len=vec_len, rand_seed=None, force_new_vecs=True,
                                 vsa_type=vsa_type, bits_per_slot=1024)
else:
    role_vecs = create_role_data(data_files=None, vec_len=10000, rand_seed=123, vsa_type=vsa_type, force_new_vecs=True,)

vsa_tok = VsaTokenizer(role_vecs, _usechunksforwords=False,
                       allow_skip_words=False, skip_words={},
                       skip_word_criterion=lambda w: False)


fr24 = json.loads(open('data/json_samples/flightradar24_track1.json').read())

encoder = FlightRadarChunkedEncoder(vsa_tok, vsa_type=vsa_type, chunk_size=80)

chunks, dbg = encoder.encode_chunks(fr24)
print(dbg)
print("num chunks:", len(chunks))

# Query: key=value should be compared against ALL chunk vectors
path = ("airline","code","icao")
value = "DLH"
q = encoder.query_key_value_raw(("airline","code","icao"), "DLH")
sims = [hsim(q, c) for c in chunks]
print("best sim:", max(sims), "chunk idx:", int(np.argmax(sims)))


tests = [
    ("airline.code.icao", ("airline","code","icao"), 8),
    ("statusDetails.squawk", ("statusDetails","squawk"), 8),
    ("statusDetails.transponder", ("statusDetails","transponder"), 16),
    ("flightHistory.departure.airport.name", ("flightHistory","departure","airport","name"), 64),
]

for name, path, ml in tests:
    val, info = encoder.decode_string_value_from_chunks_topk(
        chunks, path,
        k=4,
        max_len=ml,
        min_sim_char=0.53,
        # min_sim_char=0.525,
        stop_after_misses=4,
        min_sim_kv_verify=0.56,
        # min_sim_kv_verify=0.53,
    )
    print(f"{name}: {val!r}  info={info}")