#include "ninfer/ops/gdn_gating_proj_f16.h"
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
constexpr int kH = 48;
constexpr int kK = 5120;

int run_case(int T, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f), wd(-0.1f, 0.1f), ad(-1.0f, 1.0f);
    std::vector<__half> x(static_cast<std::size_t>(kK)*T), a(static_cast<std::size_t>(kH)*kK), b(static_cast<std::size_t>(kH)*kK);
    std::vector<float> A(kH), dt(kH);
    for (auto& v : x) v = __float2half_rn(xd(rng));
    for (auto& v : a) v = __float2half_rn(wd(rng));
    for (auto& v : b) v = __float2half_rn(wd(rng));
    for (auto& v : A) v = ad(rng);
    for (auto& v : dt) v = ad(rng);
    GuardedDeviceBuffer dx(x.size()*2), da(a.size()*2), db(b.size()*2), dA(kH*4), ddt(kH*4), dg(static_cast<std::size_t>(kH)*T*4), dbe(static_cast<std::size_t>(kH)*T*4);
    dx.copy_from_host(x.data(), dx.bytes()); da.copy_from_host(a.data(), da.bytes()); db.copy_from_host(b.data(), db.bytes());
    dA.copy_from_host(A.data(), dA.bytes()); ddt.copy_from_host(dt.data(), ddt.bytes());
    Tensor xt(dx.data(),DType::FP16,{kK,T}), at(da.data(),DType::FP16,{kH,kK}), bt(db.data(),DType::FP16,{kH,kK});
    Tensor At(dA.data(),DType::FP32,{kH}), dtt(ddt.data(),DType::FP32,{kH});
    Tensor gt(dg.data(),DType::FP32,{kH,T}), betat(dbe.data(),DType::FP32,{kH,T});
    ops::gdn_gating_proj_f16(xt, at, bt, At, dtt, gt, betat, nullptr);
    cuda_synchronize();
    const auto gg = from_device<float>(dg.data(), static_cast<std::size_t>(kH)*T);
    const auto gb = from_device<float>(dbe.data(), static_cast<std::size_t>(kH)*T);
    double e2=0, r2=0;
    for (int t=0;t<T;++t) for (int h=0;h<kH;++h) {
        double av=0,bv=0;
        for (int k=0;k<kK;++k) {
            const double xv=__half2float(x[static_cast<std::size_t>(t)*kK+k]);
            av += __half2float(a[static_cast<std::size_t>(h)*kK+k])*xv;
            bv += __half2float(b[static_cast<std::size_t>(h)*kK+k])*xv;
        }
        const double sp = av+dt[h] > 20.0 ? av+dt[h] : std::log1p(std::exp(av+dt[h]));
        const double rg = -std::exp(A[h])*sp;
        const double rb = 1.0/(1.0+std::exp(-bv));
        const std::size_t i=static_cast<std::size_t>(t)*kH+h;
        e2 += (gg[i]-rg)*(gg[i]-rg) + (gb[i]-rb)*(gb[i]-rb);
        r2 += rg*rg + rb*rb;
    }
    const double rel=std::sqrt(e2/std::max(r2,1e-30));
    std::cout<<"gdn_gating_proj_f16 T="<<T<<" rel="<<rel<<'\n';
    return rel>5e-4;
}
} // namespace
int main(){ if(cuda_unavailable()){std::cout<<"SKIP\n";return 77;} try{int f=0;f+=run_case(1,8001U);f+=run_case(4,8002U);std::cout<<(f?"FAIL":"OK")<<" gdn_gating_proj_f16\n";return f?1:0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;} }
