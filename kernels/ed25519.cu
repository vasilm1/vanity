#include "ed25519.h"

// Ed25519 prime: 2^255 - 19
#define P25519 ((int64_t)(1ULL << 255) - 19)

// Ed25519 base point coordinates
__device__ const ge25519 ed25519_base_point = {
    {{15112221, 25750622, 31903693, 50091674, 39388808, 49215580, 18968231, 11404486, 2289203, 4557935}},
    {{46316835, 65992904, 46899923, 62511023, 59091353, 36763158, 23309977, 17717912, 39069785, 13789141}},
    {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{46827403, 36985302, 46370269, 17000004, 47524426, 9209607, 62270572, 32769668, 12119206, 244841}}
};

__device__ void fe25519_reduce(fe25519 *r) {
    int32_t carry;
    int i;
    
    for (i = 0; i < 10; i++) {
        carry = r->v[i] >> 26;
        r->v[i] &= 0x3ffffff;
        if (i < 9) {
            r->v[i + 1] += carry;
        } else {
            r->v[0] += 19 * carry;
        }
    }
    
    carry = r->v[0] >> 26;
    r->v[0] &= 0x3ffffff;
    r->v[1] += carry;
}

__device__ void fe25519_mul(fe25519 *r, const fe25519 *a, const fe25519 *b) {
    int64_t t[19] = {0};
    int i, j;
    
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 10; j++) {
            t[i + j] += (int64_t)a->v[i] * b->v[j];
        }
    }
    
    for (i = 10; i < 19; i++) {
        t[i - 10] += 19 * t[i];
    }
    
    for (i = 0; i < 10; i++) {
        r->v[i] = (int32_t)t[i];
    }
    
    fe25519_reduce(r);
}

__device__ void fe25519_add(fe25519 *r, const fe25519 *a, const fe25519 *b) {
    int i;
    for (i = 0; i < 10; i++) {
        r->v[i] = a->v[i] + b->v[i];
    }
    fe25519_reduce(r);
}

__device__ void fe25519_sub(fe25519 *r, const fe25519 *a, const fe25519 *b) {
    int i;
    for (i = 0; i < 10; i++) {
        r->v[i] = a->v[i] - b->v[i] + 0x7ffffed; // Add 2*p to avoid underflow
    }
    fe25519_reduce(r);
}

__device__ void fe25519_copy(fe25519 *r, const fe25519 *a) {
    int i;
    for (i = 0; i < 10; i++) {
        r->v[i] = a->v[i];
    }
}

__device__ void fe25519_invert(fe25519 *r, const fe25519 *a) {
    // Implement Fermat's little theorem: a^(p-2) = a^(-1) (mod p)
    // For Ed25519: p-2 = 2^255 - 21
    fe25519 t;
    int i;
    
    fe25519_copy(&t, a);
    
    // Simple square-and-multiply for inversion (simplified)
    for (i = 0; i < 254; i++) {
        fe25519_mul(&t, &t, &t);  // Square
        if (i != 253) {  // Don't multiply on last iteration
            fe25519_mul(&t, &t, a);  // Multiply
        }
    }
    
    fe25519_copy(r, &t);
}

__device__ void ge25519_double(ge25519 *r, const ge25519 *p) {
    fe25519 A, B, C, D, E, F, G, H;
    
    fe25519_mul(&A, &p->X, &p->X);  // A = X1^2
    fe25519_mul(&B, &p->Y, &p->Y);  // B = Y1^2
    fe25519_mul(&C, &p->Z, &p->Z);  // C = Z1^2
    fe25519_add(&C, &C, &C);        // C = 2*Z1^2
    
    fe25519_copy(&D, &A);           // D = A
    fe25519_add(&E, &p->X, &p->Y);  // E = X1+Y1
    fe25519_mul(&E, &E, &E);        // E = (X1+Y1)^2
    fe25519_sub(&E, &E, &A);        // E = E-A
    fe25519_sub(&E, &E, &B);        // E = E-B
    
    fe25519_add(&G, &A, &B);        // G = A+B
    fe25519_sub(&F, &G, &C);        // F = G-C
    fe25519_sub(&H, &A, &B);        // H = A-B
    
    fe25519_mul(&r->X, &E, &F);     // X3 = E*F
    fe25519_mul(&r->Y, &G, &H);     // Y3 = G*H
    fe25519_mul(&r->T, &E, &H);     // T3 = E*H
    fe25519_mul(&r->Z, &F, &G);     // Z3 = F*G
}

__device__ void ge25519_add(ge25519 *r, const ge25519 *p, const ge25519 *q) {
    fe25519 A, B, C, D, E, F, G, H;
    
    fe25519_mul(&A, &p->X, &q->X);  // A = X1*X2
    fe25519_mul(&B, &p->Y, &q->Y);  // B = Y1*Y2
    fe25519_mul(&C, &p->T, &q->T);  // C = T1*T2
    fe25519_mul(&D, &p->Z, &q->Z);  // D = Z1*Z2
    
    fe25519_add(&E, &p->X, &p->Y);  // E = X1+Y1
    fe25519_add(&F, &q->X, &q->Y);  // F = X2+Y2
    fe25519_mul(&E, &E, &F);        // E = (X1+Y1)*(X2+Y2)
    fe25519_sub(&E, &E, &A);        // E = E-A
    fe25519_sub(&E, &E, &B);        // E = E-B
    
    fe25519_sub(&F, &D, &C);        // F = D-C
    fe25519_add(&G, &D, &C);        // G = D+C
    fe25519_sub(&H, &B, &A);        // H = B-A
    
    fe25519_mul(&r->X, &E, &F);     // X3 = E*F
    fe25519_mul(&r->Y, &G, &H);     // Y3 = G*H
    fe25519_mul(&r->T, &E, &H);     // T3 = E*H
    fe25519_mul(&r->Z, &F, &G);     // Z3 = F*G
}

__device__ void ge25519_scalarmult(ge25519 *r, const uint8_t *scalar, const ge25519 *base) {
    ge25519 result = {{{0}}, {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}}, {{0}}};
    ge25519 temp;
    int i, j;
    
    fe25519_copy(&temp, base);
    
    // Double-and-add scalar multiplication
    for (i = 0; i < 32; i++) {
        for (j = 0; j < 8; j++) {
            if ((scalar[i] >> j) & 1) {
                ge25519_add(&result, &result, &temp);
            }
            ge25519_double(&temp, &temp);
        }
    }
    
    *r = result;
}

__device__ void ge25519_pack(uint8_t *r, const ge25519 *p) {
    fe25519 x, y, z_inv;
    int i;
    
    // Convert to affine coordinates: (X/Z, Y/Z)
    fe25519_invert(&z_inv, &p->Z);
    fe25519_mul(&x, &p->X, &z_inv);
    fe25519_mul(&y, &p->Y, &z_inv);
    
    fe25519_reduce(&y);
    
    // Pack Y coordinate with X sign bit
    for (i = 0; i < 32; i++) {
        r[i] = 0;
    }
    
    // Simple packing (this is a simplified version)
    for (i = 0; i < 10; i++) {
        int byte_pos = (i * 26) / 8;
        int bit_pos = (i * 26) % 8;
        if (byte_pos < 32) {
            r[byte_pos] |= (y.v[i] & 0xff) << bit_pos;
            if (byte_pos + 1 < 32) {
                r[byte_pos + 1] |= (y.v[i] >> (8 - bit_pos)) & 0xff;
            }
        }
    }
    
    // Set sign bit based on X coordinate parity
    fe25519_reduce(&x);
    if (x.v[0] & 1) {
        r[31] |= 0x80;
    }
}

__device__ void ed25519_keypair(uint8_t *public_key, const uint8_t *private_key) {
    ge25519 public_point;
    
    // Perform scalar multiplication: public_key = private_key * base_point
    ge25519_scalarmult(&public_point, private_key, &ed25519_base_point);
    
    // Pack the public key
    ge25519_pack(public_key, &public_point);
} 