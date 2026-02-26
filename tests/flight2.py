import json

from vsapy import Laiho
from vsapy.role_vectors import create_role_data
from vsapy.vsa_tokenizer import VsaTokenizer
from vsapy.vsatype import VsaType
from vsapy.vsapy import hsim

from vsapy.flight_radar_vsa import FlightRadarVsaEncoder

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

encoder = FlightRadarVsaEncoder(vsa_tok, vsa_type=vsa_type)

fr24 = json.loads(open('data/json_samples/flightradar24_track1.json').read())
aircraft_vec, dbg = encoder.encode(fr24)
print(dbg)

# 1) key presence query
path = ("statusDetails", "heading")
q1 = encoder.query_key_present(path)
# q1 = encoder.query_key_present(("statusDetails", "poo"))
print(f"hsim(key present): {hsim(q1, aircraft_vec)} = {path}")

# 2) key=value query
path = ("airline", "code", "icao")
value = "DLH"
q2 = encoder.query_key_value(path, value)
print(f"hsim(key=value): {hsim(q2, aircraft_vec)} = {path}:{value}")

path = ("statusDetails", "squawk")
value = "4453"
q2 = encoder.query_key_value(path, value)
print(f"hsim(key=value): {hsim(q2, aircraft_vec)} = {path}:{value}")

path = ("statusDetails", "transponder")
value = "******"
q2 = encoder.query_key_value(path, value)
print(f"hsim(key=value): {hsim(q2, aircraft_vec)} = {path}:{value}")
path = ("statusDetails", "transponder")
value = "*ode-*"
q2 = encoder.query_key_value(path, value)
print(f"hsim(key=value): {hsim(q2, aircraft_vec)} = {path}:{value}")

path = ("statusDetails", "transponder")
value = "Mode-S"
q2 = encoder.query_key_value(path, value)
print(f"hsim(key=value): {hsim(q2, aircraft_vec)} = {path}:{value}")


path = ("flightHistory", "departure", "airport", "name")
value = "Frankfurt am Main Airport"
q2 = encoder.query_key_value(path, value)
print(f"hsim(key=value): {hsim(q2, aircraft_vec)} = {path}:{value}")

# 3) weighted query (Tern shines here; BSC will repeat)
q = encoder.query_weighted([
    (encoder.query_key_value(("airline", "code", "icao"), "DLH"), 3.0),
    (encoder.query_key_present(("flightHistory", "arrival", "airport", "code", "iata")), 1.0),
])
print("hsim(weighted):", hsim(q, aircraft_vec))


print("\n--- decode attempts ---")

path = ("airline", "code", "icao")
print(path, "=>", encoder.decode_string_value(aircraft_vec, path, max_len=8))

path = ("statusDetails", "squawk")
print(path, "=>", encoder.decode_string_value(aircraft_vec, path, max_len=8))

path = ("statusDetails", "transponder")
print(path, "=>", encoder.decode_string_value(aircraft_vec, path, max_len=12))

path = ("flightHistory", "departure", "airport", "name")
print(path, "=>", encoder.decode_string_value(aircraft_vec, path, max_len=40, min_sim=0.52, stop_after_misses=4))