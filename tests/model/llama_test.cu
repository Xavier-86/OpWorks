// Model-level tests against tests/fixtures/llama/model_fixtures.pack.
// Usage: llama_model_test <weights.pack> <model_fixtures> [longctx] [regression]
// Exits 77 (CTest skip) when the fixture files are missing.

#include "model_test_body.cuh"

int main(int argc, char **argv) {
    return modeltest::run_model_tests<opworks::LlamaModel>(argc, argv, /*bf16_tolerances=*/false);
}
