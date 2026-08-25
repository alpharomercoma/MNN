// Standalone diagnostic: run vae_decoder.mnn on a uniform (constant) input via the CPU backend,
// dumping every op's output stats (min/max/mean/std) as it executes. A uniform input through a
// correct decoder should keep every intermediate spatially-uniform per channel (min==max per
// channel-slab, or at least a tiny std). The first op where std blows up relative to its input
// is the one introducing the dot-grid/stripe artifact.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace MNN;

static void statsOf(const Tensor* t, float& mn, float& mx, float& mean, float& stddev) {
    std::shared_ptr<Tensor> host(new Tensor(t, Tensor::CAFFE, true));
    t->copyToHostTensor(host.get());
    Tensor* h = host.get();
    if (h->getType().code != halide_type_float) { mn = mx = mean = stddev = NAN; return; }
    auto ptr = h->host<float>();
    int n = h->elementSize();
    if (n <= 0 || !ptr) { mn = mx = mean = stddev = NAN; return; }
    mn = ptr[0]; mx = ptr[0];
    double sum = 0;
    for (int i = 0; i < n; ++i) { sum += ptr[i]; if (ptr[i] < mn) mn = ptr[i]; if (ptr[i] > mx) mx = ptr[i]; }
    mean = sum / n;
    double sq = 0;
    for (int i = 0; i < n; ++i) { double d = ptr[i] - mean; sq += d * d; }
    stddev = std::sqrt(sq / n);
}

// Per-channel spatial std, maxed over channels: the real signal for positional corruption.
// A uniform input pushed through a correct network should be spatially constant WITHIN each
// channel (differing bias/scale per channel is normal and not a bug), so this should be ~0 until
// the exact op that breaks spatial uniformity.
static float maxPerChannelSpatialStd(const Tensor* t) {
    std::shared_ptr<Tensor> host(new Tensor(t, Tensor::CAFFE, true));
    t->copyToHostTensor(host.get());
    Tensor* h = host.get();
    if (h->getType().code != halide_type_float) return NAN;
    auto dims = h->shape();
    if (dims.size() != 4) return NAN;
    int n = dims[0], c = dims[1], hh = dims[2], ww = dims[3];
    int spatial = hh * ww;
    if (spatial <= 1) return 0.0f;
    auto ptr = h->host<float>();
    float worst = 0.0f;
    // NCHW is the reported logical shape; MNN's actual host buffer for CPU is laid out NCHW here
    // since Tensor::createHostTensorFromDevice with a caffe-format device tensor returns caffe
    // (NCHW) host layout.
    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            const float* base = ptr + ((size_t)ni * c + ci) * spatial;
            double sum = 0;
            for (int i = 0; i < spatial; ++i) sum += base[i];
            double mean = sum / spatial;
            double sq = 0;
            for (int i = 0; i < spatial; ++i) { double d = base[i] - mean; sq += d * d; }
            float sd = std::sqrt(sq / spatial);
            if (sd > worst) worst = sd;
        }
    }
    return worst;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: vae_bisect_tool <vae_decoder.mnn> <constant_latent_value> [dump_op_name_substring]\n");
        return 1;
    }
    const char* modelPath = argv[1];
    float latentValue = atof(argv[2]);
    const char* dumpOp = argc > 3 ? argv[3] : nullptr;

    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(modelPath));
    if (!net) { printf("failed to load %s\n", modelPath); return 1; }
    net->setSessionMode(Interpreter::Session_Debug);

    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    if (::getenv("VAE_BISECT_NO_WINOGRAD")) {
        net->setSessionHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
        printf("WINOGRAD_MEMORY_LEVEL forced to 0\n");
    }
    Session* session = net->createSession(config);
    if (!session) { printf("failed to create session\n"); return 1; }

    Tensor* input = net->getSessionInput(session, nullptr);
    net->resizeTensor(input, {1, 32, 16, 16});
    net->resizeSession(session);

    {
        auto ptr = input->host<float>();
        int n = input->elementSize();
        for (int i = 0; i < n; ++i) ptr[i] = latentValue;
        printf("input: n=%d value=%f\n", n, latentValue);
    }

    TensorCallBackWithInfo before = [](const std::vector<Tensor*>& t, const OperatorInfo* info) {
        return true;
    };
    TensorCallBackWithInfo after = [dumpOp](const std::vector<Tensor*>& outs, const OperatorInfo* info) {
        for (size_t i = 0; i < outs.size(); ++i) {
            float mn, mx, mean, sd;
            statsOf(outs[i], mn, mx, mean, sd);
            float spatialSd = maxPerChannelSpatialStd(outs[i]);
            auto dims = outs[i]->shape();
            printf("[%s] type=%s out%zu shape=[", info->name().c_str(), info->type().c_str(), i);
            for (auto d : dims) printf("%d,", d);
            printf("] min=%f max=%f mean=%f std=%f maxPerChanSpatialStd=%f\n", mn, mx, mean, sd, spatialSd);

            if (dumpOp && info->name().find(dumpOp) != std::string::npos && dims.size() == 4) {
                std::shared_ptr<Tensor> host(new Tensor(outs[i], Tensor::CAFFE, true));
                outs[i]->copyToHostTensor(host.get());
                auto ptr = host->host<float>();
                int c = dims[1], hh = dims[2], ww = dims[3];
                int spatial = hh * ww;
                // Find the worst (highest spatial-std) channel, not just channel 0 -- channel 0
                // is not necessarily representative of whatever channel drives the tensor's
                // extreme min/max.
                int worstCi = 0;
                float worstSd = -1.0f;
                for (int ci = 0; ci < c; ++ci) {
                    const float* base = ptr + (size_t)ci * spatial;
                    double sum = 0;
                    for (int k = 0; k < spatial; ++k) sum += base[k];
                    double mean = sum / spatial;
                    double sq = 0;
                    for (int k = 0; k < spatial; ++k) { double d = base[k] - mean; sq += d * d; }
                    float sd = std::sqrt(sq / spatial);
                    if (sd > worstSd) { worstSd = sd; worstCi = ci; }
                }
                printf("  worst channel = %d (spatialStd=%f), grid (%dx%d):\n", worstCi, worstSd, hh, ww);
                {
                    const float* wbase = ptr + (size_t)worstCi * spatial;
                    for (int y = 0; y < hh; ++y) {
                        printf("   ");
                        for (int x = 0; x < ww; ++x) printf("%8.4f", wbase[y * ww + x]);
                        printf("\n");
                    }
                }
                printf("  first 8 channel means (channel c's constant-if-correct value):\n   ");
                for (int ci = 0; ci < std::min(c, 8); ++ci) {
                    const float* base = ptr + (size_t)ci * spatial;
                    double sum = 0;
                    for (int k = 0; k < spatial; ++k) sum += base[k];
                    printf("%8.4f", sum / spatial);
                }
                printf("\n");
                if (::getenv("VAE_DUMP_ALL_CHANNEL_MEANS")) {
                    printf("  ALL %d channel means:\n", c);
                    for (int ci = 0; ci < c; ++ci) {
                        const float* base = ptr + (size_t)ci * spatial;
                        double sum = 0;
                        for (int k = 0; k < spatial; ++k) sum += base[k];
                        printf("    ch%d: %f\n", ci, sum / spatial);
                    }
                }
            }
        }
        return true;
    };

    net->runSessionWithCallBackInfo(session, before, after, true);

    auto outputs = net->getSessionOutputAll(session);
    for (auto& kv : outputs) {
        float mn, mx, mean, sd;
        statsOf(kv.second, mn, mx, mean, sd);
        printf("FINAL OUTPUT [%s] min=%f max=%f mean=%f std=%f\n", kv.first.c_str(), mn, mx, mean, sd);
    }
    return 0;
}
