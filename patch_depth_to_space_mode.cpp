// Patches every OpType_DepthToSpace op's DepthSpaceParam.mode in a .mnn file from CRD to DCR
// (or vice versa, or to a specific target), using the flatbuffers object (*T) API: UnPack the
// whole Net into a native NetT tree, mutate, re-Pack, write out.
//
// Root cause this addresses: Sana Edit V2's vae_decoder.mnn has 5 DepthToSpace ops (one per
// up_blocks.N.0 upsample stage) all exported with mode=CRD (1). Empirically, on a uniform/smooth
// test latent, every one of these ops turns a smooth input into a garbage striped output (odd
// sub-block rows read from a wildly wrong source channel range) -- exactly what you'd get if the
// model's actual channel layout is DCR but the exported mode attribute says CRD. This patches
// mode -> DCR (0) and re-saves.
#include "MNN_generated.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>

using namespace MNN;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        printf("Usage: patch_depth_to_space_mode <in.mnn> <out.mnn> <target_mode 0=DCR|1=CRD> [name_substring_filter]\n");
        return 1;
    }
    int targetMode = atoi(argv[3]);
    const char* nameFilter = argc > 4 ? argv[4] : nullptr;

    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    auto net = GetNet(buf.data());
    std::unique_ptr<NetT> netT(net->UnPack());

    int patched = 0;
    for (auto& opPtr : netT->oplists) {
        if (opPtr->type != OpType_DepthToSpace) continue;
        if (nameFilter && opPtr->name.find(nameFilter) == std::string::npos) continue;
        auto param = opPtr->main.AsDepthSpaceParam();
        if (!param) continue;
        printf("op name=%s current mode=%d -> %d\n", opPtr->name.c_str(), (int)param->mode, targetMode);
        param->mode = (DepthToSpaceMode)targetMode;
        patched++;
    }
    printf("patched %d DepthToSpace op(s)\n", patched);

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = Net::Pack(builder, netT.get());
    builder.Finish(offset);

    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    out.close();
    printf("wrote %s (%u bytes)\n", argv[2], builder.GetSize());
    return 0;
}
