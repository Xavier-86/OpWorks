// BF16-weight model tests: same fixtures and coverage as llama_model_test,
// with the plan's BF16 tolerances (NRMSE<=1e-2, cosine>=0.999) and margin-
// analyzed generation divergences.
// Usage: llama_bf16_test <bf16_weights.pack> <model_fixtures> [longctx] [regression]

#include "model_test_body.cuh"

int main(int argc, char **argv) {
    return modeltest::run_model_tests<opworks::LlamaModelBF16>(argc, argv, /*bf16_tolerances=*/true);
}
