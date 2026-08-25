// Dumps the Region descriptors (src/dst View offset+stride, size, origin) for the
// extraTensorDescribe entry of any tensor whose name contains a given substring, by parsing
// the flatbuffer NetT directly. Used to inspect Raster ops' assembly plan without needing a
// live session.
#include "MNN_generated.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>

using namespace MNN;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: dump_regions <model.mnn> <tensor_name_substring>\n");
        return 1;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto net = GetNet(buf.data());

    // Find matching tensor indices by name.
    std::vector<int> matches;
    for (int i = 0; i < (int)net->tensorName()->size(); ++i) {
        auto n = net->tensorName()->Get(i);
        if (n && std::string(n->c_str()).find(argv[2]) != std::string::npos) {
            matches.push_back(i);
        }
    }
    printf("found %zu matching tensor(s)\n", matches.size());

    bool hasDescribe = net->extraTensorDescribe() != nullptr;
    if (!hasDescribe) {
        printf("net has no extraTensorDescribe (regions are computed at runtime, not stored statically)\n");
    }

    for (int idx : matches) {
        if (!hasDescribe) continue;
        printf("=== tensor idx=%d name=%s ===\n", idx, net->tensorName()->Get(idx)->c_str());
        bool found = false;
        for (int j = 0; j < (int)net->extraTensorDescribe()->size(); ++j) {
            auto td = net->extraTensorDescribe()->Get(j);
            if (td->index() != idx) continue;
            found = true;
            if (!td->regions()) { printf("  (no regions)\n"); continue; }
            printf("  %u region(s):\n", td->regions()->size());
            for (int r = 0; r < (int)td->regions()->size(); ++r) {
                auto reg = td->regions()->Get(r);
                printf("  region[%d]: origin=%d\n", r, reg->origin());
                if (reg->size()) {
                    printf("    size=[");
                    for (int k = 0; k < (int)reg->size()->size(); ++k) printf("%d,", reg->size()->Get(k));
                    printf("]\n");
                }
                if (reg->src()) {
                    printf("    src: offset=%d stride=[", reg->src()->offset());
                    if (reg->src()->stride()) for (int k = 0; k < (int)reg->src()->stride()->size(); ++k) printf("%d,", reg->src()->stride()->Get(k));
                    printf("]\n");
                }
                if (reg->dst()) {
                    printf("    dst: offset=%d stride=[", reg->dst()->offset());
                    if (reg->dst()->stride()) for (int k = 0; k < (int)reg->dst()->stride()->size(); ++k) printf("%d,", reg->dst()->stride()->Get(k));
                    printf("]\n");
                }
            }
        }
        if (!found) printf("  (no extraTensorDescribe entry for this tensor index)\n");
    }

    // Also print the producing op for each matched tensor, with its input tensor shapes/names,
    // so we know exactly what feeds the Raster and can reason about source layout.
    for (int idx : matches) {
        for (int i = 0; i < (int)net->oplists()->size(); ++i) {
            auto op = net->oplists()->Get(i);
            if (!op->outputIndexes()) continue;
            bool isProducer = false;
            for (int oo = 0; oo < (int)op->outputIndexes()->size(); ++oo) {
                if (op->outputIndexes()->Get(oo) == idx) isProducer = true;
            }
            if (!isProducer) continue;
            printf("=== producer op for idx=%d: name=%s type=%d (mainType=%d) ===\n", idx, op->name() ? op->name()->c_str() : "", (int)op->type(), (int)op->main_type());
            if (op->inputIndexes()) {
                for (int ii = 0; ii < (int)op->inputIndexes()->size(); ++ii) {
                    int iidx = op->inputIndexes()->Get(ii);
                    printf("  input[%d] tensor idx=%d name=%s\n", ii, iidx, net->tensorName()->Get(iidx)->c_str());
                }
            }
            if (op->main_type() == OpParameter_Reshape) {
                auto r = op->main_as_Reshape();
                printf("  Reshape dims=[");
                if (r->dims()) for (int k = 0; k < (int)r->dims()->size(); ++k) printf("%d,", r->dims()->Get(k));
                printf("] dimType=%d\n", (int)r->dimType());
            } else if (op->main_type() == OpParameter_Permute) {
                auto p = op->main_as_Permute();
                printf("  Permute dims=[");
                if (p->dims()) for (int k = 0; k < (int)p->dims()->size(); ++k) printf("%d,", p->dims()->Get(k));
                printf("]\n");
            } else if (op->main_type() == OpParameter_DepthSpaceParam) {
                auto d = op->main_as_DepthSpaceParam();
                printf("  DepthSpaceParam blockSize=%d mode=%d (0=DCR,1=CRD)\n", d->blockSize(), (int)d->mode());
            } else if (op->main_type() == OpParameter_Blob) {
                auto b = op->main_as_Blob();
                printf("  Blob dataType=%d dims=[", (int)b->dataType());
                if (b->dims()) for (int k = 0; k < (int)b->dims()->size(); ++k) printf("%d,", b->dims()->Get(k));
                printf("] float32s(first 16)=[");
                if (b->float32s()) for (int k = 0; k < (int)b->float32s()->size() && k < 16; ++k) printf("%g,", b->float32s()->Get(k));
                printf("] int32s(first 16)=[");
                if (b->int32s()) for (int k = 0; k < (int)b->int32s()->size() && k < 16; ++k) printf("%d,", b->int32s()->Get(k));
                printf("]\n");
            }
        }
    }
    return 0;
}
