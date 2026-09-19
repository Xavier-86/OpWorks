#pragma once

// Central tuning knobs and hardware constants — every "magic number" in the
// project lives here, and other files reference these names instead.
//
// The OPWORKS_* values can be overridden at compile time:
//   nvcc -DOPWORKS_THREADS=512 -DOPWORKS_MATMUL_TILE=128 ...
// They are compile-time constants on purpose: block sizes and tile shapes are
// baked into kernel instantiations (shared memory sizing, unroll factors), so
// runtime configurability would cost registers and occupancy for no benefit.

#include <cfloat>
#include <cmath>

#ifndef OPWORKS_THREADS
#define OPWORKS_THREADS 256
#endif

#ifndef OPWORKS_NUM_WAVES
#define OPWORKS_NUM_WAVES 32
#endif

#ifndef OPWORKS_PACK_SIZE
#define OPWORKS_PACK_SIZE 4
#endif

#ifndef OPWORKS_MATMUL_TILE
#define OPWORKS_MATMUL_TILE 64
#endif

#ifndef OPWORKS_MATMUL_BLOCK
#define OPWORKS_MATMUL_BLOCK 16
#endif

#ifndef OPWORKS_MATMUL_SUBTILE
#define OPWORKS_MATMUL_SUBTILE 4
#endif

#ifndef OPWORKS_LAYERNORM_EPS
#define OPWORKS_LAYERNORM_EPS 1e-5f
#endif

namespace opworks {

// Hardware constants — fixed by the CUDA architecture, not tunable.
constexpr int kWarpSize = 32;                    // threads per warp
constexpr int kMaxThreadsPerBlock = 1024;        // launch limit
constexpr unsigned kFullWarpMask = 0xffffffffu;  // __shfl_sync mask: all lanes

// Floating-point extremes, for reduction identities and clamps.
constexpr float kInf = INFINITY;
constexpr float kNegInf = -INFINITY;
constexpr float kFloatMax = FLT_MAX;     // largest finite float
constexpr float kFloatLowest = -FLT_MAX; // most negative finite float

constexpr int kThreads = OPWORKS_THREADS;    // default block size for op launches
constexpr int kNumWaves = OPWORKS_NUM_WAVES; // resident blocks per SM for latency hiding

constexpr int kPackSize = OPWORKS_PACK_SIZE; // floats per SIMD pack in elementwise kernels

constexpr int kMatmulTile = OPWORKS_MATMUL_TILE;    // output tile edge per thread block
constexpr int kMatmulBlock = OPWORKS_MATMUL_BLOCK;  // K-slice per step; block is kMatmulBlock^2 threads
constexpr int kMatmulSub = OPWORKS_MATMUL_SUBTILE;  // per-thread sub-tile edge

constexpr float kLayerNormEps = OPWORKS_LAYERNORM_EPS; // default layer_norm epsilon

static_assert(kThreads % kWarpSize == 0 && kThreads <= kMaxThreadsPerBlock, "kThreads must be a multiple of kWarpSize and fit in a block");
static_assert(kPackSize == 1 || kPackSize == 2 || kPackSize == 4, "kPackSize must be 1, 2 or 4 (alignment must be a power of two)");
static_assert(kMatmulTile == kMatmulBlock * kMatmulSub, "kMatmulTile must equal kMatmulBlock * kMatmulSub");
static_assert(kMatmulBlock * kMatmulBlock <= kMaxThreadsPerBlock, "matmul thread block must fit in kMaxThreadsPerBlock threads");

}  // namespace opworks
