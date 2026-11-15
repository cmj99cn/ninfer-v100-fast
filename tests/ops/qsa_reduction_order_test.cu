#include "ninfer/ops/qsa.h"

#include "core/arena.h"
#include "ops/op_tester.h"
#include "ops/qsa/qsa_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim  = 256;
constexpr std::int32_t kPageSize = 64;
constexpr std::int32_t kCapacity = 512;
constexpr std::int32_t kCompress = 4;

using detail::QsaAct;
using detail::qsa_row_position;

__device__ float reference_block_reduce_sum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float reference_block_reduce_max(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

template <class A>
__global__ void reference_sparse_gqa_kernel(const A* query, const A* output_gate,
                                            const int* positions, const int* valid_columns,
                                            const int* table_rows, int tokens, int batch, int axes,
                                            int head_dim, int q_heads, int kv_heads,
                                            const int* page_table, int logical_pages,
                                            const __nv_bfloat16* k_pages, const __half* v_pages,
                                            const int* selection, const int* counts, A* out) {
    __shared__ float reduce[256];
    __shared__ float probabilities[4 * kCapacity + 3];

    const int index = blockIdx.x;
    const int h     = index % q_heads;
    const int t     = (index / q_heads) % tokens;
    const int b     = index / (q_heads * tokens);
    const int columns = valid_columns == nullptr ? tokens : valid_columns[b];
    const std::int64_t output_base =
        head_dim * (h + static_cast<std::int64_t>(q_heads) *
                            (t + static_cast<std::int64_t>(tokens) * b));

    if (t >= columns) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out[output_base + d] = QsaAct<A>::from_float(0.0F);
        }
        return;
    }

    const int position = qsa_row_position(positions, t, b, batch, axes);
    const int complete = (position + 1) / kCompress;
    const int count    = counts[t + tokens * b];
    const int tail     = position + 1 - 4 * complete;
    const int count_j  = 4 * count + tail;
    const int selection_base = 512 * (t + tokens * b);
    const int group          = q_heads / kv_heads;
    const int kv_head        = h / group;
    const float scale        = 1.0F / sqrtf(static_cast<float>(head_dim));
    const int table_row      = table_rows[b];

    if (count_j == 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out[output_base + d] = QsaAct<A>::from_float(0.0F);
        }
        return;
    }

    const auto token_of = [&](int j) {
        if (j < 4 * count) { return 4 * selection[selection_base + j / 4] + j % 4; }
        return 4 * complete + (j - 4 * count);
    };
    const auto key_offset = [&](int token) {
        const int page     = token / kPageSize;
        const int row      = token % kPageSize;
        const int physical = page_table[static_cast<std::int64_t>(page) +
                                        static_cast<std::int64_t>(logical_pages) * table_row];
        return head_dim * (static_cast<std::int64_t>(row) +
                           kPageSize * (static_cast<std::int64_t>(kv_head) +
                                        static_cast<std::int64_t>(kv_heads) * physical));
    };

    for (int j = 0; j < count_j; ++j) {
        const int token = token_of(j);
        const std::int64_t cache_base = key_offset(token);
        float dot = 0.0F;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += QsaAct<A>::to_float(query[output_base + d]) *
                   __bfloat162float(k_pages[cache_base + d]);
        }
        const float total = reference_block_reduce_sum(reduce, dot);
        if (threadIdx.x == 0) { probabilities[j] = total * scale; }
    }
    __syncthreads();

    float maximum = -INFINITY;
    for (int j = threadIdx.x; j < count_j; j += blockDim.x) {
        maximum = fmaxf(maximum, probabilities[j]);
    }
    maximum = reference_block_reduce_max(reduce, maximum);

    float sum = 0.0F;
    for (int j = threadIdx.x; j < count_j; j += blockDim.x) {
        sum += expf(probabilities[j] - maximum);
    }
    const float denominator = reference_block_reduce_sum(reduce, sum);
    for (int j = threadIdx.x; j < count_j; j += blockDim.x) {
        probabilities[j] = expf(probabilities[j] - maximum) / denominator;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float attention = 0.0F;
        for (int j = 0; j < count_j; ++j) {
            const int token = token_of(j);
            const std::int64_t cache_base = key_offset(token);
            attention += probabilities[j] * __half2float(v_pages[cache_base + d]);
        }
        const float gate = QsaAct<A>::to_float(output_gate[output_base + d]);
        const float sigmoid = 1.0F / (1.0F + expf(-gate));
        out[output_base + d] = QsaAct<A>::from_float(attention * sigmoid);
    }
}

Tensor tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape) {
    return Tensor(data, dtype, shape);
}

std::vector<float> random_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    return values;
}

struct Case {
    const char* label = "";
    DType activation  = DType::BF16;
    std::int32_t T    = 0;
    std::int32_t B    = 1;
    std::int32_t axes = 1;
    std::uint32_t seed = 0;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> valid;
    std::vector<std::int32_t> table_rows;
    std::vector<std::int32_t> page_table;
    std::vector<std::int32_t> selection;
    std::vector<std::int32_t> counts;
    std::int32_t logical_pages  = 1;
    std::int32_t physical_pages = 1;
    std::int32_t q_heads        = 24;
    std::int32_t kv_heads       = 2;
};

int run_case(const Case& c, cudaStream_t stream, WorkspaceArena& arena) {
    const std::size_t act_count =
        static_cast<std::size_t>(kHeadDim) * c.q_heads * c.T * c.B;
    const std::size_t page_count =
        static_cast<std::size_t>(kHeadDim) * kPageSize * c.kv_heads * c.physical_pages;

    DeviceBuffer d_query =
        c.activation == DType::FP16
            ? to_device_f16(random_values(act_count, c.seed, -2.0F, 2.0F))
            : to_device_bf16(random_values(act_count, c.seed, -2.0F, 2.0F));
    DeviceBuffer d_gate =
        c.activation == DType::FP16
            ? to_device_f16(random_values(act_count, c.seed + 1, -2.0F, 2.0F))
            : to_device_bf16(random_values(act_count, c.seed + 1, -2.0F, 2.0F));
    DeviceBuffer d_k = to_device_bf16(random_values(page_count, c.seed + 2, -2.0F, 2.0F));
    DeviceBuffer d_v = to_device_f16(random_values(page_count, c.seed + 3, -2.0F, 2.0F));
    DeviceBuffer d_positions = to_device_i32(c.positions);
    DeviceBuffer d_valid     = to_device_i32(c.valid);
    DeviceBuffer d_rows      = to_device_i32(c.table_rows);
    DeviceBuffer d_page      = to_device_i32(c.page_table);
    DeviceBuffer d_selection = to_device_i32(c.selection);
    DeviceBuffer d_counts    = to_device_i32(c.counts);
    DeviceBuffer d_out(act_count * 2);
    DeviceBuffer d_ref(act_count * 2);
    d_out.fill(0);
    d_ref.fill(0);

    const std::int32_t table_rows = static_cast<std::int32_t>(c.table_rows.size());
    QsaCacheView cache{
        tensor(d_k.p, DType::BF16, {kHeadDim, kPageSize, c.kv_heads, c.physical_pages}),
        tensor(d_v.p, DType::FP16, {kHeadDim, kPageSize, c.kv_heads, c.physical_pages}),
        Tensor{},
        tensor(d_page.p, DType::I32, {c.logical_pages, table_rows})};
    QsaBatchView batch{tensor(d_positions.p, DType::I32, {c.T, c.axes, c.B}),
                       tensor(d_valid.p, DType::I32, {c.B}),
                       tensor(d_rows.p, DType::I32, {c.B})};
    QsaBlockSelection selected{tensor(d_selection.p, DType::I32, {kCapacity, c.T, c.B}),
                               tensor(d_counts.p, DType::I32, {c.T, c.B})};

    const int grid        = c.T * c.B * c.q_heads;
    const auto* positions = static_cast<const int*>(d_positions.p);
    const auto* valid     = static_cast<const int*>(d_valid.p);
    const auto* rows      = static_cast<const int*>(d_rows.p);
    const auto* page      = static_cast<const int*>(d_page.p);
    const auto* selection = static_cast<const int*>(d_selection.p);
    const auto* counts    = static_cast<const int*>(d_counts.p);
    if (c.activation == DType::FP16) {
        reference_sparse_gqa_kernel<__half><<<grid, 256, 0, stream>>>(
            static_cast<const __half*>(d_query.p), static_cast<const __half*>(d_gate.p),
            positions, valid, rows, c.T, c.B, c.axes, kHeadDim, c.q_heads, c.kv_heads, page,
            c.logical_pages, static_cast<const __nv_bfloat16*>(d_k.p),
            static_cast<const __half*>(d_v.p), selection, counts, static_cast<__half*>(d_ref.p));
    } else {
        reference_sparse_gqa_kernel<__nv_bfloat16><<<grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(d_query.p),
            static_cast<const __nv_bfloat16*>(d_gate.p), positions, valid, rows, c.T, c.B,
            c.axes, kHeadDim, c.q_heads, c.kv_heads, page, c.logical_pages,
            static_cast<const __nv_bfloat16*>(d_k.p), static_cast<const __half*>(d_v.p),
            selection, counts, static_cast<__nv_bfloat16*>(d_ref.p));
    }
    cuda_check_last_launch("reference_sparse_gqa_kernel launch");

    Tensor out_view = tensor(d_out.p, c.activation, {kHeadDim, c.q_heads, c.T, c.B});
    ops::qsa_sparse_gqa(
        tensor(d_query.p, c.activation, {kHeadDim, c.q_heads, c.T, c.B}),
        tensor(d_gate.p, c.activation, {kHeadDim, c.q_heads, c.T, c.B}), cache, batch, selected,
        static_cast<std::uint32_t>(c.T), arena, out_view, stream);
    cuda_synchronize(stream);

    const std::vector<std::uint16_t> ref = from_device<std::uint16_t>(d_ref, act_count);
    const std::vector<std::uint16_t> got = from_device<std::uint16_t>(d_out, act_count);
    std::size_t diffs = 0;
    for (std::size_t i = 0; i < act_count; ++i) {
        if (ref[i] != got[i]) { ++diffs; }
    }
    if (diffs != 0) {
        std::cerr << "qsa_reduction_order_test: " << c.label << " differs on " << diffs << " of "
                  << act_count << " activation words\n";
        return 1;
    }
    std::cout << "qsa_reduction_order_test: " << c.label << " exact (" << act_count << " words)\n";
    return 0;
}

std::vector<std::int32_t> complete_of(const std::vector<std::int32_t>& positions, std::int32_t T,
                                      std::int32_t B, std::int32_t axes) {
    std::vector<std::int32_t> complete(static_cast<std::size_t>(T) * B, 0);
    for (std::int32_t b = 0; b < B; ++b) {
        for (std::int32_t t = 0; t < T; ++t) {
            const std::int32_t p = positions[static_cast<std::size_t>(t * axes * B + b)];
            complete[static_cast<std::size_t>(t + T * b)] = (p + 1) / kCompress;
        }
    }
    return complete;
}

std::vector<std::int32_t> identity_selection(std::int32_t T, std::int32_t B,
                                             const std::vector<std::int32_t>& complete) {
    std::vector<std::int32_t> selection(static_cast<std::size_t>(kCapacity) * T * B, 0);
    for (std::int32_t b = 0; b < B; ++b) {
        for (std::int32_t t = 0; t < T; ++t) {
            const std::size_t base    = static_cast<std::size_t>(kCapacity) * (t + T * b);
            const std::int32_t blocks = complete[static_cast<std::size_t>(t + T * b)];
            for (std::int32_t g = 0; g < blocks; ++g) { selection[base + g] = g; }
        }
    }
    return selection;
}

std::vector<std::int32_t> permutation_pages(std::int32_t logical_pages,
                                            std::int32_t table_rows) {
    std::vector<std::int32_t> page(static_cast<std::size_t>(logical_pages) * table_rows, 0);
    for (std::int32_t r = 0; r < table_rows; ++r) {
        for (std::int32_t l = 0; l < logical_pages; ++l) {
            page[static_cast<std::size_t>(l + logical_pages * r)] = (l * 3 + r) % logical_pages;
        }
    }
    return page;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    DeviceArena arena(1024);

    int failures = 0;

    {
        Case c{.label = "prefill tails bf16", .activation = DType::BF16, .T = 8, .B = 1,
               .axes = 1, .seed = 100, .positions = {0, 1, 2, 3, 8, 10, 13, 7},
               .valid = {8}, .table_rows = {0}, .logical_pages = 2, .physical_pages = 2,
               .q_heads = 24, .kv_heads = 2};
        c.counts     = complete_of(c.positions, c.T, c.B, c.axes);
        c.selection  = identity_selection(c.T, c.B, c.counts);
        c.page_table = permutation_pages(c.logical_pages, 1);
        failures += run_case(c, stream, arena);
    }

    {
        Case c{.label = "prefill tails fp16 batch2 axes3", .activation = DType::FP16, .T = 8,
               .B = 2, .axes = 3, .seed = 200, .valid = {8, 5}, .table_rows = {0, 1},
               .logical_pages = 4, .physical_pages = 4, .q_heads = 24, .kv_heads = 2};
        c.positions.resize(static_cast<std::size_t>(c.T) * c.axes * c.B);
        const std::vector<std::int32_t> axis0_a{0, 1, 2, 3, 8, 10, 13, 7};
        const std::vector<std::int32_t> axis0_b{11, 14, 15, 19, 23, 27, 31, 35};
        for (std::int32_t b = 0; b < c.B; ++b) {
            for (std::int32_t t = 0; t < c.T; ++t) {
                const std::int32_t p = (b == 0 ? axis0_a : axis0_b)[static_cast<std::size_t>(t)];
                for (std::int32_t a = 0; a < c.axes; ++a) {
                    c.positions[static_cast<std::size_t>((t * c.axes + a) * c.B + b)] = p + a;
                }
            }
        }
        c.counts     = complete_of(c.positions, c.T, c.B, c.axes);
        c.selection  = identity_selection(c.T, c.B, c.counts);
        c.page_table = permutation_pages(c.logical_pages, 2);
        failures += run_case(c, stream, arena);
    }

    {
        Case c{.label = "partial selection bf16", .activation = DType::BF16, .T = 3, .B = 1,
               .axes = 1, .seed = 300, .positions = {40, 41, 42}, .valid = {3},
               .table_rows = {0}, .counts = {2, 3, 5}, .logical_pages = 2,
               .physical_pages = 2, .q_heads = 24, .kv_heads = 2};
        const std::int32_t ids[3][5] = {{1, 4, -1, -1, -1}, {0, 2, 7, -1, -1}, {3, 5, 6, 9, 8}};
        const std::int32_t sizes[3]  = {2, 3, 5};
        c.selection.assign(static_cast<std::size_t>(kCapacity) * c.T, 0);
        for (std::int32_t t = 0; t < c.T; ++t) {
            for (std::int32_t i = 0; i < sizes[t]; ++i) {
                c.selection[static_cast<std::size_t>(kCapacity) * t + i] = ids[t][i];
            }
        }
        c.page_table = permutation_pages(c.logical_pages, 1);
        failures += run_case(c, stream, arena);
    }

    {
        Case c{.label = "empty set bf16", .activation = DType::BF16, .T = 2, .B = 1, .axes = 1,
               .seed = 400, .positions = {3, 7}, .valid = {2}, .table_rows = {0},
               .counts = {0, 0}, .logical_pages = 1, .physical_pages = 1, .q_heads = 24,
               .kv_heads = 2};
        c.selection.assign(static_cast<std::size_t>(kCapacity) * c.T, 0);
        c.page_table = permutation_pages(c.logical_pages, 1);
        failures += run_case(c, stream, arena);
    }

    {
        Case c{.label = "tail only fp16", .activation = DType::FP16, .T = 2, .B = 1, .axes = 1,
               .seed = 500, .positions = {4, 9}, .valid = {2}, .table_rows = {0},
               .counts = {0, 0}, .logical_pages = 1, .physical_pages = 1, .q_heads = 24,
               .kv_heads = 2};
        c.selection.assign(static_cast<std::size_t>(kCapacity) * c.T, 0);
        c.page_table = permutation_pages(c.logical_pages, 1);
        failures += run_case(c, stream, arena);
    }

    {
        Case c{.label = "tp2 partial fp16", .activation = DType::FP16, .T = 3, .B = 1, .axes = 1,
               .seed = 600, .positions = {9, 18, 27}, .valid = {3}, .table_rows = {0},
               .counts = {1, 2, 0}, .logical_pages = 2, .physical_pages = 2, .q_heads = 12,
               .kv_heads = 1};
        c.selection.assign(static_cast<std::size_t>(kCapacity) * c.T, 0);
        c.selection[static_cast<std::size_t>(kCapacity) * 0 + 0] = 1;
        c.selection[static_cast<std::size_t>(kCapacity) * 1 + 0] = 1;
        c.selection[static_cast<std::size_t>(kCapacity) * 1 + 1] = 2;
        c.page_table = permutation_pages(c.logical_pages, 1);
        failures += run_case(c, stream, arena);
    }

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    if (failures != 0) {
        std::cerr << "qsa_reduction_order_test: " << failures << " case(s) failed\n";
        return 1;
    }
    std::cout << "qsa_reduction_order_test: PASS\n";
    return 0;
}
