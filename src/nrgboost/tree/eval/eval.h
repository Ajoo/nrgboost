#include <stdint.h>

typedef struct {
    uint16_t right;
    uint8_t dim;
    uint8_t value;
} Node32;

typedef struct {
    int16_t left;
    int16_t right;
    uint16_t dim;
    uint16_t value;
} Node64;

typedef struct {
    uint16_t start;
    uint16_t stop;
} Range64;

typedef struct {
    uint16_t max_value;
    uint8_t* mask;
} Set64;

typedef struct {
    int num_dims;
    int* is_dim_ordered;
    void** support;
} ProductDomain;

typedef struct {
    ProductDomain* domain;
    Node64* nodes;
    double* values;
    int num_nodes;
    int num_leaves;
} Tree64;

void ProductDomain_print(const ProductDomain* domain);

void Tree64_print(const Tree64* tree);

void Tree64_eval_uint8(const Tree64* tree, int16_t* output, const uint8_t* data, int num_samples); 

void Tree64_eval_slice_uint8(const Tree64* tree, int16_t* output, const uint8_t* data, int num_samples, int slice_dim);

void cumsumexp(double* logits, int num_values);

int sample_cat(double u, double* probs, int num_values);

void Tree64_gibbs_uint8(
    const Tree64** trees, 
    int num_trees, 
    uint8_t* output, 
    int num_samples, 
    uint8_t* sample, 
    const double** initial_logits, 
    double temperature,
    const uint64_t* seed_seq
    );

void Tree64_gibbs_uint8_p(
    const Tree64** trees,
    int num_trees,
    uint8_t* output,
    int num_samples_per_chain,
    int num_chains,
    const uint8_t* samples,
    const double** initial_logits,
    double temperature,
    const uint64_t* seed_seq,
    int num_parallel
    );

void bincount_p(
    const uint8_t* data,
    int num_dims,
    int num_samples,
    int64_t** counts,
    int num_parallel
    );

void weighted_bincount_p(
    const uint8_t* data,
    const double* weights,
    int num_dims,
    int num_samples,
    int64_t** counts,
    int num_parallel
    );

void sample32(double* output, int num_samples, uint64_t seed);

void sample64(double* output, int num_samples, uint64_t seed);
