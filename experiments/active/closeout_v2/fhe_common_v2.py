"""Shared CKKS helpers for closeout_v2 (TenSEAL; verified against the installed library).

Two TenSEAL behaviours matter here and were measured, not assumed (see test_v2.py):

1. `CKKSVector.mm` relies on the input vector's replicated slot layout. Adding a bias as
   a plain Python list breaks that layout for a FOLLOWING `mm`: the error jumps from
   ~1e-5 to ~1e-2 on a 192->768->192 chain. Biases are therefore added as fresh
   encryptions under the public key (ciphertext + ciphertext, zero levels).
2. Scalars, `square`, `polyval` and scalar multiplication preserve the layout.

The server works on a copy of the client context made public with
`make_context_public()`, so it holds no secret key. Only ciphertexts are serialized,
which avoids the >2 GB context-serialization failure seen earlier with Galois keys.
"""
import time

import numpy as np


def contexts(ts, ring_degree, modulus_bits, scale_bits, threads=4, galois=True):
    """Client context (secret key) and a public server copy. SEAL rejects chains that
    exceed its 128-bit (tc128) modulus allowance for the ring degree."""
    client = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=ring_degree,
                        coeff_mod_bit_sizes=list(modulus_bits), n_threads=threads)
    client.global_scale = 2 ** scale_bits
    if galois:
        client.generate_galois_keys()
    server = client.copy()
    server.make_context_public()
    if server.has_secret_key():
        raise RuntimeError('server context still holds the secret key')
    return client, server


def level(v):
    return v.ciphertext()[0].coeff_modulus_size()


def to_server(ts, server, v):
    return ts.ckks_vector_from(server, v.serialize())


def to_client(ts, client, v):
    return ts.ckks_vector_from(client, v.serialize())


def mlp_weights(fc1_w, fc1_b, fc2_w, fc2_b, gelu_poly):
    """Fold the polynomial's input scaling into fc1: z = (h W1^T + b1 - c)/r."""
    c, r = gelu_poly['center'], gelu_poly['radius']
    return dict(A=np.asarray(fc1_w, np.float64).T / r, a=(np.asarray(fc1_b, np.float64) - c) / r,
                coefficients=list(gelu_poly['coefficients']), W2T=np.asarray(fc2_w, np.float64).T,
                b2=np.asarray(fc2_b, np.float64))


def plaintext_mlp(h, x_mid, w):
    """Float64 reference of exactly the encrypted arithmetic: x_mid + fc2(p(fc1-scaled(h)))."""
    z = h @ w['A'] + w['a']
    g = np.polynomial.polynomial.polyval(z, w['coefficients'])
    return x_mid + g @ w['W2T'] + w['b2']


def encrypted_mlp(ts, client, server, h, x_mid, w):
    """One token: Enc(h), Enc(x_mid) -> server computes x_mid + fc2(p(fc1(h))) -> Dec.
    No intermediate decryption. Returns output and a cost record."""
    t0 = time.perf_counter()
    eh = to_server(ts, server, ts.ckks_vector(client, list(map(float, h))))
    ex = to_server(ts, server, ts.ckks_vector(client, list(map(float, x_mid))))
    start_level = level(eh)
    t1 = time.perf_counter()
    z = eh.mm(w['A'].tolist()) + ts.ckks_vector(server, w['a'].tolist())
    g = z.polyval(w['coefficients'])
    o = g.mm(w['W2T'].tolist()) + ts.ckks_vector(server, w['b2'].tolist())
    out = o + ex
    t2 = time.perf_counter()
    end_level = level(out)
    y = np.asarray(to_client(ts, client, out).decrypt(), dtype=np.float64)
    t3 = time.perf_counter()
    return y, dict(encrypt_seconds=t1 - t0, server_seconds=t2 - t1, decrypt_seconds=t3 - t2,
                   levels_consumed=start_level - end_level, start_primes=start_level, end_primes=end_level,
                   intermediate_decryptions=0)
