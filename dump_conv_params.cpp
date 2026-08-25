// Dumps Convolution2D params (kernel/stride/pad/dilation/group/channels) for every op in a .mnn
// file whose name contains a given substring, by parsing the flatbuffer NetT directly.
#include "MNN_generated.h"
#include <cstdio>
#include <fstream>
#include <vector>

using namespace MNN;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: dump_conv_params <model.mnn> <name_substring>\n");
        return 1;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto net = GetNet(buf.data());
    for (int i = 0; i < (int)net->oplists()->size(); ++i) {
        auto op = net->oplists()->Get(i);
        std::string name = op->name() ? op->name()->str() : "";
        if (name.find(argv[2]) == std::string::npos) continue;
        printf("op[%d] name=%s type=%d\n", i, name.c_str(), (int)op->type());
        if (op->main_type() == OpParameter_Convolution2D) {
            auto conv = op->main_as_Convolution2D();
            auto common = conv->common();
            printf("  kernel=(%d,%d) stride=(%d,%d) pad=(%d,%d) padMode=%d dilate=(%d,%d) group=%d inputCount=%d outputCount=%d relu=%d relu6=%d\n",
                common->kernelX(), common->kernelY(), common->strideX(), common->strideY(),
                common->padX(), common->padY(), (int)common->padMode(),
                common->dilateX(), common->dilateY(), common->group(),
                common->inputCount(), common->outputCount(), common->relu(), common->relu6());
            if (conv->weight()) printf("  weight size=%u\n", conv->weight()->size());
            if (conv->bias()) printf("  bias size=%u\n", conv->bias()->size());
        }
        for (int ii = 0; ii < (int)op->inputIndexes()->size(); ++ii) {
            int idx = op->inputIndexes()->Get(ii);
            printf("  input[%d] tensor idx=%d name=%s\n", ii, idx, net->tensorName()->Get(idx)->c_str());
        }
        for (int oo = 0; oo < (int)op->outputIndexes()->size(); ++oo) {
            int idx = op->outputIndexes()->Get(oo);
            printf("  output[%d] tensor idx=%d name=%s\n", oo, idx, net->tensorName()->Get(idx)->c_str());
        }
    }
    return 0;
}
