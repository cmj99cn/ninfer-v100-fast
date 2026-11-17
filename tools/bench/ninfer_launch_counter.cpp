#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Stat {
    long long count = 0;
    double api_ms = 0.0;
};

std::map<std::string, Stat> g_raw;
std::map<std::string, Stat> g_family;
std::map<const void*, std::string> g_kernel_names;
long long g_total = 0;
double g_api_ms = 0.0;

std::int64_t ns_now() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

const char* family_of(const std::string& n) {
    struct Entry { const char* key; const char* label; };
    static const Entry table[] = {
        {"moe_nvfp4", "moe_routed"},
        {"gate_up", "moe_routed"},
        {"routed_down", "moe_routed"},
        {"moe_shared", "moe_shared"},
        {"route_probe", "moe_route"},
        {"moe_nvfp4_route", "moe_route"},
        {"gdn_projection", "gdn_proj"},
        {"gdn_input", "gdn_proj"},
        {"causal_conv", "gdn_conv"},
        {"gated_delta", "gdn_recurrent"},
        {"delta_net", "gdn_recurrent"},
        {"gated_rmsnorm", "gdn_norm"},
        {"hc_", "hc"},
        {"gated_residual", "hc"},
        {"sparse_gqa", "qsa"},
        {"qsa_", "qsa"},
        {"indexer", "qsa"},
        {"linear_", "linear"},
        {"gemm", "linear"},
        {"volta_mma", "linear"},
        {"w8_", "linear"},
        {"extract_bf16", "extract"},
        {"argmax", "misc"},
        {"embedding", "misc"},
        {"hc_expand", "misc"},
    };
    for (const auto& e : table) {
        if (n.find(e.key) != std::string::npos) { return e.label; }
    }
    return "other";
}

std::string kernel_name(const void* func) {
    if (func == nullptr) { return "<null>"; }
    auto it = g_kernel_names.find(func);
    if (it != g_kernel_names.end()) {
        int status = 0;
        char* dem = abi::__cxa_demangle(it->second.c_str(), nullptr, nullptr, &status);
        std::string name = (status == 0 && dem != nullptr) ? dem : it->second;
        if (dem != nullptr) { std::free(dem); }
        auto paren = name.find('(');
        if (paren != std::string::npos) { name = name.substr(0, paren); }
        if (name.size() > 160) { name = name.substr(0, 160); }
        return name;
    }
    Dl_info info{};
    if (dladdr(func, &info) == 0 || info.dli_sname == nullptr) { return "<unknown>"; }
    int status = 0;
    char* dem = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
    std::string name = (status == 0 && dem != nullptr) ? dem : info.dli_sname;
    if (dem != nullptr) { std::free(dem); }
    auto paren = name.find('(');
    if (paren != std::string::npos) { name = name.substr(0, paren); }
    if (name.size() > 160) { name = name.substr(0, 160); }
    return name;
}

void record(const void* func, double api_ms) {
    std::string name = kernel_name(func);
    g_raw[name].count += 1;
    g_raw[name].api_ms += api_ms;
    std::string fam = family_of(name);
    g_family[fam].count += 1;
    g_family[fam].api_ms += api_ms;
    g_total += 1;
    g_api_ms += api_ms;
}

} // namespace

namespace {
bool gate_open() {
    const char* value = std::getenv("NINFER_LAUNCH_GATE");
    return value != nullptr && value[0] == '1';
}
} // namespace

using LaunchFn = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
using LaunchExFn = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
using RegisterFn = void (*)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*,
                            dim3*, int*);

extern "C" void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                                        const char* deviceName, int threadLimit, uint3* tid,
                                        uint3* bid, dim3* bDim, dim3* gDim, int* wSize) {
    static RegisterFn real = nullptr;
    if (real == nullptr) {
        real = reinterpret_cast<RegisterFn>(dlsym(RTLD_NEXT, "__cudaRegisterFunction"));
    }
    if (hostFun != nullptr && deviceName != nullptr) {
        g_kernel_names[hostFun] = deviceName;
    }
    real(fatCubinHandle, hostFun, deviceFun, deviceName, threadLimit, tid, bid, bDim, gDim, wSize);
}

extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args,
                                        std::size_t smem, cudaStream_t stream) {
    static LaunchFn real = nullptr;
    if (real == nullptr) { real = reinterpret_cast<LaunchFn>(dlsym(RTLD_NEXT, "cudaLaunchKernel")); }
    if (!gate_open()) { return real(func, grid, block, args, smem, stream); }
    const std::int64_t begin = ns_now();
    cudaError_t result = real(func, grid, block, args, smem, stream);
    const double ms = static_cast<double>(ns_now() - begin) / 1.0e6;
    record(func, ms);
    return result;
}

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func,
                                           void** args) {
    static LaunchExFn real = nullptr;
    if (real == nullptr) { real = reinterpret_cast<LaunchExFn>(dlsym(RTLD_NEXT, "cudaLaunchKernelExC")); }
    if (!gate_open()) { return real(config, func, args); }
    const std::int64_t begin = ns_now();
    cudaError_t result = real(config, func, args);
    const double ms = static_cast<double>(ns_now() - begin) / 1.0e6;
    record(func, ms);
    return result;
}

extern "C" __attribute__((destructor)) void ninfer_launch_dump() {
    const char* path = std::getenv("NINFER_LAUNCH_LOG");
    if (path == nullptr || path[0] == '\0') { return; }
    FILE* out = std::fopen(path, "w");
    if (out == nullptr) { return; }
    std::fprintf(out, "total_launches %lld\napi_ms %.4f\n\n[family]\n", g_total, g_api_ms);
    std::vector<std::pair<std::string, Stat>> fam(g_family.begin(), g_family.end());
    std::sort(fam.begin(), fam.end(), [](const auto& a, const auto& b) {
        return a.second.count > b.second.count;
    });
    for (const auto& [name, stat] : fam) {
        std::fprintf(out, "%-16s count %6lld  api_ms %8.4f\n", name.c_str(), stat.count, stat.api_ms);
    }
    std::fprintf(out, "\n[raw top 70]\n");
    std::vector<std::pair<std::string, Stat>> raw(g_raw.begin(), g_raw.end());
    std::sort(raw.begin(), raw.end(), [](const auto& a, const auto& b) {
        return a.second.count > b.second.count;
    });
    for (std::size_t i = 0; i < raw.size() && i < 70; ++i) {
        std::fprintf(out, "%4lld  api %7.4f  %s\n", raw[i].second.count, raw[i].second.api_ms,
                     raw[i].first.c_str());
    }
    std::fclose(out);
}
