#ifndef ED25519_H
#define ED25519_H

#include <stdint.h>

// Ed25519 field element (10 limbs of 26/25 bits each)
typedef struct {
    int32_t v[10];
} fe25519;

// Ed25519 point in extended coordinates (X:Y:Z:T)
typedef struct {
    fe25519 X, Y, Z, T;
} ge25519;

// Ed25519 base point
__device__ extern const ge25519 ed25519_base_point;

// Field arithmetic
__device__ void fe25519_reduce(fe25519 *r);
__device__ void fe25519_mul(fe25519 *r, const fe25519 *a, const fe25519 *b);
__device__ void fe25519_add(fe25519 *r, const fe25519 *a, const fe25519 *b);
__device__ void fe25519_sub(fe25519 *r, const fe25519 *a, const fe25519 *b);
__device__ void fe25519_copy(fe25519 *r, const fe25519 *a);
__device__ void fe25519_invert(fe25519 *r, const fe25519 *a);

// Point operations
__device__ void ge25519_double(ge25519 *r, const ge25519 *p);
__device__ void ge25519_add(ge25519 *r, const ge25519 *p, const ge25519 *q);
__device__ void ge25519_scalarmult(ge25519 *r, const uint8_t *scalar, const ge25519 *base);
__device__ void ge25519_pack(uint8_t *r, const ge25519 *p);

// Main function to generate keypair
__device__ void ed25519_keypair(uint8_t *public_key, const uint8_t *private_key);

#endif 