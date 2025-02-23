#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <pcg_variants.h>
#include "eval.h"

#define MAX_CARDINALITY 256
#define NUM_BYTES(x) (x / 8 + (x % 8 != 0))
#define BIT(i) (0x80 >> (i%8))
#define TO_UNIT_INTERVAL_64(i) (ldexp(i, -64))
// #define TO_UNIT_INTERVAL(i) (double) i/RAND_MAX

Range64 get_ordered_support(const ProductDomain* domain, int dim) {
    return *((Range64 *) domain->support[dim]);
}

Set64 get_unordered_support(const ProductDomain* domain, int dim) {
    return *((Set64 *) domain->support[dim]);
}

uint16_t get_ordered_cardinality(const ProductDomain* domain, int dim) {
    return ((Range64 *) domain->support[dim])->stop;
}

uint16_t get_unordered_cardinality(const ProductDomain* domain, int dim) {
    return ((Set64 *) domain->support[dim])->max_value;
}

void ProductDomain_print(const ProductDomain* domain){
    Range64 range;
    Set64 set;

    for (int i=0; i<domain->num_dims; i++) {
        if (domain->is_dim_ordered[i]) {
            range = get_ordered_support(domain, i);
            printf("%d: Range(%d, %d)\n", i, range.start, range.stop);
        } else {
            set = get_unordered_support(domain, i);
            printf("%d: Set(%02X", i, set.mask[0]);
            for (int j=1; j<NUM_BYTES(set.max_value); j++) {
                printf(", %02X", set.mask[j]);
            }
            printf(")\n");
        }
    }
}

void Tree64_print(const Tree64* tree)
{
    Node64 node;
    for (int k=0; k<tree->num_nodes; k++) {
        node = tree->nodes[k];
        printf("%d: left = %d, right = %d, dim = %d, value = %d\n", k, node.left, node.right, node.dim, node.value); 
    }
}

// TODO: output should probably be uint8_t
void Tree64_eval_uint8(const Tree64* tree, int16_t* output, const uint8_t* data, int num_samples)
{
    int k;
    int num_dims = tree->domain->num_dims;
    int* is_dim_ordered = tree->domain->is_dim_ordered;
    uint16_t dim;
    Node64* root = tree->nodes;
    Node64* node;
    

    for (int i=0; i<num_samples; i++) {
        k = 0;
        while (k >= 0) {
            node = root + k;
            dim = node->dim;
            assert(dim < num_dims);

            if (is_dim_ordered[dim] ? (data[i*num_dims + dim] >= node->value) : (data[i*num_dims + dim] == node->value)) {
                k = node->right;
            } else {
                k = node->left;
            };
            assert(k < tree->num_nodes);
        };
        output[i] = ~k;
    }
}

void Tree64_eval_ordered_slice_sample_uint8(
    const Tree64* tree, 
    int16_t* output,
    uint16_t* node_stack, 
    Range64* range_stack,
    const uint8_t* sample, 
    int slice_dim
    )
{
    int sp = -1;
    int i;
    uint16_t dim;
    Node64* node;
    
    int16_t current_node = 0;

    int* is_dim_ordered = tree->domain->is_dim_ordered;
    Range64 current_range = get_ordered_support(tree->domain, slice_dim);

    while (1) {
        // printf("sp: %d      ", sp);
        // printf("current node: %d\n", current_node);

        if (current_node < 0) { /* terminal */
            // printf("terminal node!\n");
            // printf("start: %d, stop: %d\n", current_range.start, current_range.stop);
            current_node = ~current_node;
            for (i=current_range.start; i<current_range.stop; i++)
                output[i] = current_node;

            if (sp < 0)
                break;

            current_node = node_stack[sp];
            current_range = range_stack[sp];
            sp--; /* branch is over, pop from stack */
            continue;
        }

        node = tree->nodes + current_node;
        dim = node->dim;
        // printf("node dim: %d\n", dim);
        // assert(dim < (tree->domain).num_dims);

        if (dim == slice_dim) { /* evaluate two branches */
            // printf("slice dim!\n");
            // printf("node value: %d\n", node->value);
            sp++;
            node_stack[sp] = node->right;
            range_stack[sp] = (Range64){.start=node->value, .stop=current_range.stop};

            current_node = node->left;
            current_range.stop = node->value;
            continue;
        } 

        if (is_dim_ordered[dim] ? (sample[dim] >= node->value) : (sample[dim] == node->value)) {
            // printf("go right!\n");
            current_node = node->right;
        } else {
            // printf("go left!\n");
            current_node = node->left;
        };
        assert(current_node < tree->num_nodes);
    }
}


void Tree64_eval_unordered_slice_sample_uint8(
    const Tree64* tree, 
    int16_t* output,
    uint16_t* node_stack, 
    uint8_t* cat_stack,
    const uint8_t* sample, 
    int slice_dim
    )
{
    int sp = -1;
    int i;
    int byte;
    uint16_t dim;
    uint8_t mask;
    Node64* node;

    int* is_dim_ordered = tree->domain->is_dim_ordered;
    Set64 current_set = get_unordered_support(tree->domain, slice_dim);
    int num_bytes = NUM_BYTES(current_set.max_value); //NUM_BYTES(num_values);

    int16_t current_node = 0;
    uint8_t* current_mask = cat_stack;
    // Reserve first position for mask of current node
    memcpy(current_mask, current_set.mask, num_bytes);
    cat_stack += num_bytes;
    
    while (1) {
        // printf("sp: %d      ", sp);
        // printf("current node: %d\n", current_node);

        if (current_node < 0) { /* terminal */
            // printf("terminal node!\n");
            current_node = ~current_node;
            // for (i=0; i<num_bytes; i++)
            //     printf("bitpacked values[%d]: %d\n", i, current_mask[i]);
            for (i=0; i<current_set.max_value; i++) {
                if (BIT(i) & current_mask[i/8])
                    output[i] = current_node;
            }
            /* branch is over, pop from stack */
            if (sp < 0)
                break;
            current_node = node_stack[sp];
            current_mask = cat_stack + sp*num_bytes;
            sp--; 
            continue;
        }

        node = tree->nodes + current_node;
        dim = node->dim;
        // printf("node dim: %d\n", dim);
        // assert(dim < (tree->domain).num_dims);

        if (dim == slice_dim) { /* evaluate two branches */
            // printf("slice dim!\n");
            // printf("node value: %d\n", node->value);
            /* right branch */
            mask = BIT(node->value);
            byte = node->value / 8;

            sp++;
            node_stack[sp] = node->right;
            memset(cat_stack + sp*num_bytes, 0, num_bytes);
            cat_stack[sp*num_bytes + byte] = mask;

            /* left branch */
            current_node = node->left;
            current_mask[byte] &= ~mask;

            // for (i=0; i<num_bytes; i++)
            //     printf("left bitpacked values[%d]: %d\n", i, current_mask[i]);

            // for (i=0; i<num_bytes; i++)
            //     printf("right bitpacked values[%d]: %d\n", i, cat_stack[sp*num_bytes + i]);
            continue;
        } 

        if (is_dim_ordered[dim] ? (sample[dim] >= node->value) : (sample[dim] == node->value)) {
            // printf("go right!\n");
            current_node = node->right;
        } else {
            // printf("go left!\n");
            current_node = node->left;
        };
        assert(current_node < tree->num_nodes);
    }
}


void Tree64_eval_unordered_slice_sample_composite_uint8(
    const Tree64* tree, 
    int16_t* output,
    uint16_t* node_stack, 
    uint8_t* cat_stack,
    const uint8_t* sample, 
    int slice_dim
    )
{
    int sp = -1;
    int i;
    uint16_t dim;
    Node64* node;

    int* is_dim_ordered = tree->domain->is_dim_ordered;
    Set64 current_set = get_unordered_support(tree->domain, slice_dim);
    int num_bytes = NUM_BYTES(current_set.max_value); //NUM_BYTES(num_values);

    int16_t right_node;
    int16_t current_node = 0;
    uint8_t* right_mask;
    // Reserve first position for mask of current node
    memcpy(cat_stack, current_set.mask, num_bytes);
    uint8_t* current_mask = cat_stack;
    cat_stack += num_bytes;
    
    while (1) {
        // printf("sp: %d      ", sp);
        // printf("current node: %d\n", current_node);

        if (current_node < 0) { /* terminal */
            // printf("terminal node!\n");
            current_node = ~current_node;
            // for (i=0; i<num_bytes; i++)
            //    printf("bitpacked values[%d]: %d\n", i, current_mask[i]);
            for (i=0; i<current_set.max_value; i++) {
                if (BIT(i) & current_mask[i/8])
                    output[i] = current_node;
            }
            /* branch is over, pop from stack */
            if (sp < 0)
                break;
            current_node = node_stack[sp];
            memcpy(current_mask, cat_stack + sp*num_bytes, num_bytes);
            sp--; 
            continue;
        }

        node = tree->nodes + current_node;
        dim = node->dim;
        // printf("node dim: %d\n", dim);
        //assert(dim < tree->domain.num_dims);

        if (dim == slice_dim) { /* evaluate two branches */
            // printf("slice dim!\n");
            sp++; 
            current_node = node->left;
            right_node = node->right;

            node_stack[sp] = right_node;
            right_mask = cat_stack + sp*num_bytes;
            
            memset(right_mask, 0, num_bytes);

            right_mask[node->value / 8] |= BIT(node->value);
            while ((current_node > 0) && ((tree->nodes + current_node)->right == right_node)) {
                node = tree->nodes + current_node;
                current_node = node->left;
                right_mask[node->value / 8] |= BIT(node->value);
            }
            for (i=0; i<num_bytes; i++)
                current_mask[i] &= ~right_mask[i];
            // for (i=0; i<num_bytes; i++)
            //     printf("left bitpacked values[%d]: %d\n", i, current_mask[i]);
            // for (i=0; i<num_bytes; i++)
            //     printf("right bitpacked values[%d]: %d\n", i, right_mask[i]);
            continue;
        }

        if (is_dim_ordered[dim] ? (sample[dim] >= node->value) : (sample[dim] == node->value)) {
            // printf("go right!\n");
            current_node = node->right;
        } else {
            // printf("go left!\n");
            current_node = node->left;
        };
        assert(current_node < tree->num_nodes);
    }
}


void Tree64_eval_ordered_slice_uint8(
    const Tree64* tree, 
    int16_t* output,
    const uint8_t* data, 
    int num_samples,
    int slice_dim
    )
{
    /* TODO: improve this */
    int max_stack = tree->num_nodes;
    int num_dims = tree->domain->num_dims;
    int num_output_values = get_ordered_support(tree->domain, slice_dim).stop;

    uint16_t* node_stack = (uint16_t*) malloc(max_stack * sizeof(uint16_t)); 
    Range64* range_stack = (Range64*) malloc(max_stack * sizeof(Range64));

    for (int i=0; i<num_samples; i++) {
        Tree64_eval_ordered_slice_sample_uint8(
            tree,
            output + i*num_output_values,
            node_stack,
            range_stack,
            data + i*num_dims,
            slice_dim
        );
    }

    free(node_stack);
    free(range_stack);
}


void Tree64_eval_unordered_slice_uint8(
    const Tree64* tree, 
    int16_t* output,
    const uint8_t* data, 
    int num_samples,
    int slice_dim
    )
{
    // printf("Evaluating unordered slice\n");
    /* TODO: improve this */
    int max_stack = tree->num_nodes;
    int num_output_values = get_unordered_support(tree->domain, slice_dim).max_value;
    int num_bytes = NUM_BYTES(num_output_values);
    int num_dims = tree->domain->num_dims;

    uint16_t* node_stack = (uint16_t*) malloc(max_stack * sizeof(uint16_t));
    uint8_t* cat_stack = (uint8_t*) malloc(max_stack * num_bytes);

    for (int i=0; i<num_samples; i++) {
        // printf(">>>> Sample %d\n", i);
        Tree64_eval_unordered_slice_sample_composite_uint8(
            tree,
            output + i*num_output_values,
            node_stack,
            cat_stack,
            data + i*num_dims,
            slice_dim
        );
    }

    free(node_stack);
    free(cat_stack);
}


void Tree64_eval_slice_uint8(
    const Tree64* tree, 
    int16_t* output,
    const uint8_t* data, 
    int num_samples,
    int slice_dim
    )
{
    if (tree->domain->is_dim_ordered[slice_dim]){
        Tree64_eval_ordered_slice_uint8(tree, output, data, num_samples, slice_dim);
    } else {
        Tree64_eval_unordered_slice_uint8(tree, output, data, num_samples, slice_dim);
    }
}


int Tree64_max_num_nodes(const Tree64** trees, int num_trees)
{
    // TODO: improve this by computing maximum branching factor
    int max = 0;
    for (int i=0; i<num_trees; i++)
        if (trees[i]->num_nodes > max)
            max = trees[i]->num_nodes;
    return max;
}


void cumsumexp(double* logits, int num_values) {
    double max = logits[0], sumexp = 0.0;
    int i;

    for (i=1; i<num_values; i++) {
        if (logits[i] > max)
            max = logits[i];
    }

    for (i=0; i<num_values; i++) {
        sumexp += exp(logits[i] - max);
        logits[i] = sumexp;
    }

    for (i=0; i<num_values; i++)
        logits[i] /= sumexp;
}


void tempered_cumsumexp(double* logits, int num_values, double temperature) {
    double max = logits[0], sumexp = 0.0;
    int i;

    for (i=1; i<num_values; i++) {
        if (logits[i] > max)
            max = logits[i];
    }

    for (i=0; i<num_values; i++) {
        sumexp += exp( (logits[i] - max) / temperature );
        logits[i] = sumexp;
    }

    for (i=0; i<num_values; i++)
        logits[i] /= sumexp;
}

// TODO: replace linear search with something better
int sample_cat(double u, double* probs, int num_values) {
    for (int i=0; i<num_values; i++) 
        if (u < probs[i])
            return i;
    return -1;
}


void _Tree64_gibbs_uint8(
    const Tree64** trees,
    int num_trees,
    uint8_t* output,
    int num_samples,
    uint8_t* sample,
    double** initial_logits,
    double temperature,
    pcg64_random_t* rng,
    uint16_t* node_stack,
    uint8_t* val_stack
    //int16_t* leaf_idx,
    //double* logits
    )
{
    int i, j, t, k, cardinality;
    ProductDomain* domain = trees[0]->domain;
    int num_dims = domain->num_dims;
    int* is_dim_ordered = domain->is_dim_ordered;
    double u;

    int16_t leaf_idx[MAX_CARDINALITY];
    double logits[MAX_CARDINALITY]; 

    for (i=0; i<num_samples; i++) {
        for (j=0; j<num_dims; j++) {
            if (is_dim_ordered[j]) {
                cardinality = get_ordered_cardinality(domain, j);
                if (initial_logits==NULL) {
                    memset(logits, 0, cardinality * sizeof(double));
                } else {
                    memcpy(logits, initial_logits[j], cardinality * sizeof(double));
                }
                
                for (t=0; t<num_trees; t++) {
                    Tree64_eval_ordered_slice_sample_uint8(
                        trees[t],
                        leaf_idx,
                        node_stack,
                        (Range64*) val_stack,
                        sample,
                        j
                    );
                    for (k=0; k<cardinality; k++)
                        logits[k] += trees[t]->values[leaf_idx[k]];
                }
                
            } else {
                cardinality = get_unordered_cardinality(domain, j);
                if (initial_logits==NULL) {
                    memset(logits, 0, cardinality * sizeof(double));
                } else {
                    memcpy(logits, initial_logits[j], cardinality * sizeof(double));
                }

                for (t=0; t<num_trees; t++) {
                    Tree64_eval_unordered_slice_sample_composite_uint8(
                        trees[t],
                        leaf_idx,
                        node_stack,
                        val_stack,
                        sample,
                        j
                    );
                    for (k=0; k<cardinality; k++)
                        logits[k] += trees[t]->values[leaf_idx[k]];
                }
            }
            tempered_cumsumexp(logits, cardinality, temperature);

            u = TO_UNIT_INTERVAL_64(pcg64_random_r(rng));
            sample[j] = (uint8_t) sample_cat(u, logits, cardinality);
        }
        memcpy(output + i*num_dims, sample, num_dims);
    }

}


void Tree64_gibbs_uint8(
    const Tree64** trees,
    int num_trees,
    uint8_t* output,
    int num_samples,
    uint8_t* sample,
    const double** initial_logits,
    double temperature,
    const uint64_t* seed_seq
    )
{
    assert(num_trees > 0);

    // create stacks
    int max_stack = Tree64_max_num_nodes(trees, num_trees);
    // TODO: for now set to maximum possible categorical mask size
    int num_bytes = 32; //sizeof(Range64);
    //int max_cardinality = 255;

    uint16_t* node_stack = (uint16_t*) malloc(max_stack * sizeof(uint16_t)); 
    uint8_t* val_stack = (uint8_t*) malloc(max_stack * num_bytes);
    // TODO: these are fixed. can be static
    //int16_t* leaf_idx = (int16_t*) malloc(max_cardinality * sizeof(int16_t));
    //double* logits = (double*) malloc(max_cardinality * sizeof(double));

    // setup rngs
    pcg64_random_t rng;
    pcg64_srandom_r(&rng, seed_seq[0], seed_seq[1]);

    _Tree64_gibbs_uint8(
        trees, num_trees, output, num_samples, sample, initial_logits, temperature, &rng,
        node_stack, val_stack //, leaf_idx, logits
    );

    free(node_stack);
    free(val_stack);
    // free(leaf_idx);
    // free(logits);
}


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
    )
{
    assert(num_trees > 0);

    // create stacks
    int max_stack = Tree64_max_num_nodes(trees, num_trees);
    // TODO: for now set to maximum possible categorical mask size
    int num_bytes = 32; //sizeof(Range64);
    // int max_cardinality = 255;
    int num_dims = trees[0]->domain->num_dims;

    if (num_parallel <= 0)
        num_parallel = omp_get_max_threads();
    if (num_parallel > num_chains)
        num_parallel = num_chains;

    // printf("Sampling...\n");
    // for (int i=0; i<num_chains; i++)
    //    printf("Chain %d: %" PRIu64 ", %" PRIu64 "\n", i,  seed_seq[2*i], seed_seq[2*i+1]);
        
    // printf("Requesting a pool of %d threads.\n", num_parallel);
    omp_set_num_threads(num_parallel);
    #pragma omp parallel
    {
        // #pragma omp single
        // printf("Setting up resources for pool of %d threads.\n", omp_get_num_threads());

        // Setup per thread resources
        uint16_t* node_stack = (uint16_t*) malloc(max_stack * sizeof(uint16_t)); 
        uint8_t* val_stack = (uint8_t*) malloc(max_stack * num_bytes);
        uint8_t* sample = (uint8_t*) malloc(num_dims);
        // TODO: these are fixed for now. can be static
        // int16_t* leaf_idx = (int16_t*) malloc(max_cardinality * sizeof(int16_t));
        // double* logits = (double*) malloc(max_cardinality * sizeof(double));

        pcg64_random_t rng;

        #pragma omp for schedule(static) //schedule(dynamic, 1) // nowait
        for (int i=0; i<num_chains; i++) {
            // #pragma omp critical
            // printf("Running chain %d of %d on thread %d.\n", i, num_chains, omp_get_thread_num());

            // setup chain
            memcpy(sample, samples + num_dims*i, num_dims);
            pcg64_srandom_r(&rng, seed_seq[2*i], seed_seq[2*i+1]);

            _Tree64_gibbs_uint8(
                trees, num_trees, 
                output + num_dims*num_samples_per_chain*i, 
                num_samples_per_chain, 
                sample, 
                initial_logits, temperature,
                &rng,
                node_stack, val_stack //, leaf_idx, logits
            );
        }

        free(node_stack);
        free(val_stack);
        free(sample);
        // free(leaf_idx);
        // free(logits);
    }

}


void bincount_p(
    const uint8_t* data,
    int num_dims,
    int num_samples,
    int64_t** counts,
    int num_parallel
    )
{
    if (num_parallel > 0)
       omp_set_num_threads(num_parallel);

    #pragma omp parallel for
    for (int d=0; d<num_dims; d++) {
        uint8_t* col = data + d*num_samples;
        int64_t* cnt = counts[d];
        for (int i=0; i<num_samples; i++)
            cnt[col[i]]++;
    }
}


void weighted_bincount_p(
    const uint8_t* data,
    const double* weights,
    int num_dims,
    int num_samples,
    int64_t** counts,
    int num_parallel
    )
{
    if (num_parallel > 0)
        omp_set_num_threads(num_parallel);

    #pragma omp parallel for //schedule(dynamic, 1) // nowait
    for (int d=0; d<num_dims; d++) {
        uint8_t* col = data + d*num_samples;
        int64_t* cnt = counts[d];
        for (int i=0; i<num_samples; i++)
            cnt[col[i]] += weights[i];
    }
}


void sample32(double* output, int num_samples, uint64_t seed) {
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, 0, seed);

    for (int i=0; i<num_samples; i++) {
        output[i] = ldexp(pcg32_random_r(&rng), -32);
    }
    // double d = ldexp(pcg32_random_r(&myrng), -32);
}


void sample64(double* output, int num_samples, uint64_t seed) {
    pcg64_random_t rng;
    pcg64_srandom_r(&rng, 0, seed);

    for (int i=0; i<num_samples; i++) {
        output[i] = ldexp(pcg64_random_r(&rng), -64);
    }
    // double d = ldexp(pcg32_random_r(&myrng), -32);
}
