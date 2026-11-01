#include "ninfer/ops/causal_conv1d_silu_f16.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
int run_case(int C, int T, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<__half> x(static_cast<std::size_t>(C)*T), w(static_cast<std::size_t>(C)*4), st(static_cast<std::size_t>(C)*3);
    for (auto& v : x) v = __float2half_rn(d(rng));
    for (auto& v : w) v = __float2half_rn(d(rng));
    for (auto& v : st) v = __float2half_rn(d(rng));
    GuardedDeviceBuffer dx(x.size()*2), dw(w.size()*2), ds(st.size()*2), dout(x.size()*2);
    dx.copy_from_host(x.data(), dx.bytes()); dw.copy_from_host(w.data(), dw.bytes()); ds.copy_from_host(st.data(), ds.bytes());
    Tensor xt(dx.data(),DType::FP16,{C,T}), wt(dw.data(),DType::FP16,{C,4}), stt(ds.data(),DType::FP16,{3,C}), ot(dout.data(),DType::FP16,{C,T});
    ops::causal_conv1d_silu_f16(xt, wt, stt, ot, nullptr);
    cuda_synchronize();
    const auto ob = from_device<std::uint16_t>(dout.data(), x.size());
    const auto sb = from_device<std::uint16_t>(ds.data(), st.size());
    std::vector<float> oracle_out(static_cast<std::size_t>(C)*T);
    std::vector<float> state(static_cast<std::size_t>(C)*3);
    for (int i=0;i<C*3;++i) state[i]=__half2float(st[i]);
    for (int t=0;t<T;++t) for (int c=0;c<C;++c) {
        float x0 = t>=3 ? __half2float(x[static_cast<std::size_t>(t-3)*C+c]) : state[c];
        float x1 = t>=2 ? __half2float(x[static_cast<std::size_t>(t-2)*C+c]) : state[C+c];
        float x2 = t>=1 ? __half2float(x[static_cast<std::size_t>(t-1)*C+c]) : state[2*C+c];
        float x3 = __half2float(x[static_cast<std::size_t>(t)*C+c]);
        float a = x0*__half2float(w[c]) + x1*__half2float(w[C+c]) + x2*__half2float(w[2*C+c]) + x3*__half2float(w[3*C+c]);
        oracle_out[static_cast<std::size_t>(t)*C+c] = a/(1.0f+std::exp(-a));
    }
    for (int c=0;c<C;++c) {
        if (T>=3) { state[c]=__half2float(x[static_cast<std::size_t>(T-3)*C+c]); state[C+c]=__half2float(x[static_cast<std::size_t>(T-2)*C+c]); state[2*C+c]=__half2float(x[static_cast<std::size_t>(T-1)*C+c]); }
        else if (T==2) { state[c]=state[C+c]; state[C+c]=state[2*C+c]; state[2*C+c]=__half2float(x[C+c]); }
        else { state[c]=state[C+c]; state[C+c]=state[2*C+c]; state[2*C+c]=__half2float(x[c]); }
    }
    int failures=0; double e2=0,r2=0;
    for (std::size_t i=0;i<oracle_out.size();++i) { double g=__half2float(__ushort_as_half(ob[i])); double e=g-oracle_out[i]; e2+=e*e; r2+=oracle_out[i]*oracle_out[i]; }
    const double rel=std::sqrt(e2/std::max(r2,1e-30));
    for (int i=0;i<C*3;++i) if (sb[i] != __half_as_ushort(__float2half_rn(state[i]))) ++failures;
    std::cout<<"causal_conv1d_silu_f16 C="<<C<<" T="<<T<<" rel="<<rel<<" state_fail="<<failures<<'\n';
    if (rel>2e-3) ++failures;
    return failures;
}
} // namespace
int main(){ if(cuda_unavailable()){std::cout<<"SKIP\n";return 77;} try{int f=0;f+=run_case(16,1,9001U);f+=run_case(16,5,9002U);f+=run_case(17,7,9003U);std::cout<<(f?"FAIL":"OK")<<" causal_conv1d_silu_f16\n";return f?1:0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;} }
