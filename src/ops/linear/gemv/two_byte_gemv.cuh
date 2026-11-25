#pragma once

// Two-byte (BF16/FP16) row-major GEMV kernels for sm_70 decode (T == 1). Element-type generic:
// both instantiations share the pack layout, phase math, and shared-memory staging; only the
// pairwise conversion (bits <-> float2) and the final scalar round differ. The geometry and
// schedule types in ops/linear/bf16/bf16_config.h are dtype-agnostic despite their Bf16 names.

#include "ops/linear/bf16/bf16_config.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Element>
struct GemvTwoByteTraits;

template <>
struct GemvTwoByteTraits<__nv_bfloat16> {
    static __device__ __forceinline__ float2 pair_to_float2(std::uint32_t bits) {
        return bf16x2_bits_to_float2(bits);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct GemvTwoByteTraits<__half> {
    static __device__ __forceinline__ float2 pair_to_float2(std::uint32_t bits) {
        return __half22float2(load_vec<__half2>(&bits));
    }
    static __device__ __forceinline__ __half from_float(float value) {
        return __float2half_rn(value);
    }
};

template <int Values>
struct alignas(Values* static_cast<int>(sizeof(std::uint16_t))) GemvPack {
    static_assert(Values == 4 || Values == 8 || Values == 16);
    std::uint32_t words[Values / 2];
};

static_assert(sizeof(GemvPack<4>) == 8);
static_assert(sizeof(GemvPack<8>) == 16);
static_assert(sizeof(GemvPack<16>) == 32);

template <class Element, int Values>
__device__ __forceinline__ GemvPack<Values> load_gemv_pack(const Element* pointer) {
    if constexpr (Values <= 8) {
        return load_vec<GemvPack<Values>>(pointer);
    } else {
        GemvPack<Values> result;
        const GemvPack<8> low  = load_vec<GemvPack<8>>(pointer);
        const GemvPack<8> high = load_vec<GemvPack<8>>(pointer + 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            result.words[pair]     = low.words[pair];
            result.words[pair + 4] = high.words[pair];
        }
        return result;
    }
}

template <class Element, Bf16WeightCache Cache, int Values>
__device__ __forceinline__ GemvPack<Values> load_gemv_weight_pack(const Element* pointer) {
    if constexpr (Cache == Bf16WeightCache::Default) {
        return load_gemv_pack<Element, Values>(pointer);
    } else if constexpr (Values == 4) {
        uint2 bits;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                     : "=r"(bits.x), "=r"(bits.y)
                     : "l"(pointer));
        return load_vec<GemvPack<Values>>(&bits);
    } else if constexpr (Values == 8) {
        uint4 bits;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                     : "l"(pointer));
        return load_vec<GemvPack<Values>>(&bits);
    } else {
        GemvPack<Values> result;
        const GemvPack<8> low  = load_gemv_weight_pack<Element, Cache, 8>(pointer);
        const GemvPack<8> high = load_gemv_weight_pack<Element, Cache, 8>(pointer + 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            result.words[pair]     = low.words[pair];
            result.words[pair + 4] = high.words[pair];
        }
        return result;
    }
}

template <class Element, int Values, int AccumulatorChains>
__device__ __forceinline__ void accumulate_gemv_packs(const GemvPack<Values>& weight,
                                                      const GemvPack<Values>& activation,
                                                      float (&accumulators)[AccumulatorChains]) {
#pragma unroll
    for (int pair = 0; pair < Values / 2; ++pair) {
        const float2 w           = GemvTwoByteTraits<Element>::pair_to_float2(weight.words[pair]);
        const float2 x           = GemvTwoByteTraits<Element>::pair_to_float2(activation.words[pair]);
        constexpr int kChainMask = AccumulatorChains - 1;
        accumulators[(2 * pair) & kChainMask] =
            fmaf(w.x, x.x, accumulators[(2 * pair) & kChainMask]);
        accumulators[(2 * pair + 1) & kChainMask] =
            fmaf(w.y, x.y, accumulators[(2 * pair + 1) & kChainMask]);
    }
}

template <class Element>
struct GemvContiguousOutput {
    Element* data;

    __device__ __forceinline__ void store(std::int32_t parent_row, Element value) const {
        data[parent_row] = value;
    }
};

template <class Element>
struct GemvStoreEpilogue {
    template <class Output>
    __device__ __forceinline__ void operator()(const Output& output, std::int32_t parent_row,
                                               float accumulator) const {
        output.store(parent_row, GemvTwoByteTraits<Element>::from_float(accumulator));
    }
};

// Column-major [rows, Columns] output for the small-T batched decode route. Column c lives at a
// fixed stride of `rows` elements, matching out.slice(1, c, 1) of the T==1 sequential caller.
template <class Element>
struct GemvColumnContiguousOutput {
    Element* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, Element value, int column) const {
        data[static_cast<std::int64_t>(column) * rows + parent_row] = value;
    }
};

template <class Element>
struct GemvColumnStoreEpilogue {
    template <class Output>
    __device__ __forceinline__ void operator()(const Output& output, std::int32_t parent_row,
                                               float accumulator, int column) const {
        output.store(parent_row, GemvTwoByteTraits<Element>::from_float(accumulator), column);
    }
};

template <class Geometry, class Schedule, class Element>
struct GemvSharedStorage {
    static constexpr int kActivationElements =
        Schedule::kActivationAccess == Bf16ActivationAccess::Shared ? Geometry::kInputRows : 8;
    static constexpr int kReductionWarps = Schedule::kWarpsPerRow > 1 ? Schedule::kWarpsPerRow : 1;

    alignas(16) Element activation[kActivationElements];
    float partials[Schedule::kRowGroupsPerCta][Schedule::kRowsPerWarp][kReductionWarps];
};

template <class Element, class Geometry, class Schedule>
__device__ __forceinline__ const Element*
prepare_gemv_activation(const Element* x, GemvSharedStorage<Geometry, Schedule, Element>& shared) {
    if constexpr (Schedule::kActivationAccess == Bf16ActivationAccess::Direct) {
        return x;
    } else {
        static_assert((Geometry::kInputRows % 8) == 0);
        constexpr int kPacks = Geometry::kInputRows / 8;
        auto* destination    = reinterpret_cast<GemvPack<8>*>(shared.activation);
        const auto* source   = reinterpret_cast<const GemvPack<8>*>(x);
        for (int pack = static_cast<int>(threadIdx.x); pack < kPacks; pack += Schedule::kThreads) {
            destination[pack] = source[pack];
        }
        __syncthreads();
        return shared.activation;
    }
}

template <class Geometry, class Schedule>
__device__ __forceinline__ std::int32_t gemv_phase_offset(int phase, int warp_in_row, int lane) {
    constexpr int kValuesPerWarp = kWarpSize * Schedule::kValuesPerLane;
    return phase * Schedule::kWarpsPerRow * kValuesPerWarp + warp_in_row * kValuesPerWarp +
           lane * Schedule::kValuesPerLane;
}

template <class Element, class Geometry, class Schedule>
__device__ __forceinline__ GemvPack<Schedule::kValuesPerLane>
load_gemv_activation_phase(const Element* activation, int phase, int warp_in_row, int lane) {
    const int offset = gemv_phase_offset<Geometry, Schedule>(phase, warp_in_row, lane);
    return load_gemv_pack<Element, Schedule::kValuesPerLane>(activation + offset);
}

template <class Element, class Geometry, class Schedule>
__device__ __forceinline__ GemvPack<Schedule::kValuesPerLane>
load_gemv_weight_phase(const Element* weight, int row, int phase, int warp_in_row, int lane) {
    const int offset = gemv_phase_offset<Geometry, Schedule>(phase, warp_in_row, lane);
    return load_gemv_weight_pack<Element, Schedule::kWeightCache, Schedule::kValuesPerLane>(
        weight + static_cast<std::int64_t>(row) * Geometry::kInputRows + offset);
}

template <class Schedule, int Phases>
__device__ __forceinline__ int gemv_phase_index(int iteration, int row0) {
    if constexpr (Schedule::kPhaseOrder == Bf16PhaseOrder::Sequential) {
        return iteration;
    } else {
        const int row_group = row0 / Schedule::kRowsPerWarp;
        int phase           = iteration + (row_group * Schedule::kPhaseStride) % Phases;
        if (phase >= Phases) { phase -= Phases; }
        return phase;
    }
}

template <class Element, class Geometry, class Schedule>
__device__ __forceinline__ void compute_gemv_rows(
    const Element* activation, const Element* weight, int row0, int warp_in_row,
    int lane, float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains]) {
    constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * kWarpSize * Schedule::kValuesPerLane;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);
    constexpr int kPhases = Geometry::kInputRows / kValuesPerPhase;
    using Pack            = GemvPack<Schedule::kValuesPerLane>;

    if constexpr (Schedule::kPrefetchDepth == 1) {
#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < kPhases; ++iteration) {
            const int phase     = gemv_phase_index<Schedule, kPhases>(iteration, row0);
            const Pack x_values = load_gemv_activation_phase<Element, Geometry, Schedule>(activation, phase,
                                                                                         warp_in_row, lane);
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const Pack w_values = load_gemv_weight_phase<Element, Geometry, Schedule>(
                    weight, row0 + local_row, phase, warp_in_row, lane);
                accumulate_gemv_packs<Element>(w_values, x_values, accumulators[local_row]);
            }
        }
    } else {
        const int first_phase = gemv_phase_index<Schedule, kPhases>(0, row0);
        Pack current_x = load_gemv_activation_phase<Element, Geometry, Schedule>(activation, first_phase,
                                                                                 warp_in_row, lane);
        Pack current_w[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            current_w[local_row] = load_gemv_weight_phase<Element, Geometry, Schedule>(
                weight, row0 + local_row, first_phase, warp_in_row, lane);
        }

#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < kPhases; ++iteration) {
            Pack next_x;
            Pack next_w[Schedule::kRowsPerWarp];
            if (iteration + 1 < kPhases) {
                const int next_phase = gemv_phase_index<Schedule, kPhases>(iteration + 1, row0);
                next_x = load_gemv_activation_phase<Element, Geometry, Schedule>(activation, next_phase,
                                                                                 warp_in_row, lane);
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    next_w[local_row] = load_gemv_weight_phase<Element, Geometry, Schedule>(
                        weight, row0 + local_row, next_phase, warp_in_row, lane);
                }
            }
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                accumulate_gemv_packs<Element>(current_w[local_row], current_x, accumulators[local_row]);
            }
            if (iteration + 1 < kPhases) {
                current_x = next_x;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    current_w[local_row] = next_w[local_row];
                }
            }
        }
    }
}

template <class Element, class Geometry, class Schedule, class Output, class Epilogue>
__device__ __forceinline__ void gemv_body(const Element* __restrict__ x, const Element* __restrict__ weight,
                                          Output output, Epilogue epilogue) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    __shared__ GemvSharedStorage<Geometry, Schedule, Element> shared;
    const Element* activation = prepare_gemv_activation<Element, Geometry, Schedule>(x, shared);

    const int lane        = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp        = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_group   = warp / Schedule::kWarpsPerRow;
    const int warp_in_row = warp % Schedule::kWarpsPerRow;
    const int cta_row0    = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int row0        = cta_row0 + row_group * Schedule::kRowsPerWarp;
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};

    compute_gemv_rows<Element, Geometry, Schedule>(activation, weight, row0, warp_in_row, lane,
                                                   accumulators);

    float warp_totals[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        warp_totals[local_row] = warp_reduce_sum(total);
    }

    if constexpr (Schedule::kWarpsPerRow == 1) {
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                epilogue(output, row0 + local_row, warp_totals[local_row]);
            }
        }
    } else {
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                shared.partials[row_group][local_row][warp_in_row] = warp_totals[local_row];
            }
        }
        __syncthreads();
        if (warp_in_row == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const float partial = lane < Schedule::kWarpsPerRow
                                          ? shared.partials[row_group][local_row][lane]
                                          : 0.0F;
                const float total   = warp_reduce_sum(partial);
                if (lane == 0) { epilogue(output, row0 + local_row, total); }
            }
        }
    }
}

// Multi-column counterpart of gemv_body. The weight pack of each row/phase is fetched once and
// applied to every token column; each column keeps its own accumulator set and goes through the
// exact same chain accumulation, phase order, warp reduction and from_float rounding as the T==1
// path, so the result is raw-bit identical to a sequential T==1 replay.
template <class Element, class Geometry, class Schedule, int Columns, class Output, class Epilogue>
__device__ __forceinline__ void gemv_multi_body(const Element* __restrict__ x,
                                                const Element* __restrict__ weight, Output output,
                                                Epilogue epilogue) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert(Schedule::kActivationAccess == Bf16ActivationAccess::Direct);
    static_assert(Schedule::kWarpsPerRow == 1);
    static_assert(Schedule::kPrefetchDepth == 1);
    static_assert((Geometry::kInputRows % 8) == 0);

    const int lane        = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp        = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_group   = warp / Schedule::kWarpsPerRow;
    const int warp_in_row = warp % Schedule::kWarpsPerRow;
    const int row0        = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta +
                           row_group * Schedule::kRowsPerWarp;
    (void)warp_in_row;

    constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * kWarpSize * Schedule::kValuesPerLane;
    constexpr int kPhases         = Geometry::kInputRows / kValuesPerPhase;
    using Pack                    = GemvPack<Schedule::kValuesPerLane>;

    float accumulators[Columns][Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
#pragma unroll
    for (int iteration = 0; iteration < kPhases; ++iteration) {
        const int phase = gemv_phase_index<Schedule, kPhases>(iteration, row0);
        Pack w_values[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            w_values[local_row] = load_gemv_weight_phase<Element, Geometry, Schedule>(
                weight, row0 + local_row, phase, 0, lane);
        }
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const Element* column_x =
                x + static_cast<std::int64_t>(column) * Geometry::kInputRows;
            const Pack x_values =
                load_gemv_activation_phase<Element, Geometry, Schedule>(column_x, phase, 0, lane);
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                accumulate_gemv_packs<Element>(w_values[local_row], x_values,
                                               accumulators[column][local_row]);
            }
        }
    }

#pragma unroll
    for (int column = 0; column < Columns; ++column) {
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            float total = 0.0F;
#pragma unroll
            for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                total += accumulators[column][local_row][chain];
            }
            total = warp_reduce_sum(total);
            if (lane == 0) { epilogue(output, row0 + local_row, total, column); }
        }
    }
}

template <class Geometry, class Schedule, int Columns>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp16_gemv_multi_kernel(
    const __half* __restrict__ x, const __half* __restrict__ weight,
    GemvColumnContiguousOutput<__half> output) {
    gemv_multi_body<__half, Geometry, Schedule, Columns>(
        x, weight, output, GemvColumnStoreEpilogue<__half>{});
}

template <class Geometry, class Schedule, class Output, class Epilogue = GemvStoreEpilogue<__nv_bfloat16>>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight, Output output,
    Epilogue epilogue = {}) {
    gemv_body<__nv_bfloat16, Geometry, Schedule>(x, weight, output, epilogue);
}

template <class Geometry, class Schedule, class Output, class Epilogue = GemvStoreEpilogue<__half>>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp16_gemv_kernel(
    const __half* __restrict__ x, const __half* __restrict__ weight, Output output,
    Epilogue epilogue = {}) {
    gemv_body<__half, Geometry, Schedule>(x, weight, output, epilogue);
}

} // namespace ninfer::ops::detail
