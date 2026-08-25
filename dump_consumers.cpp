// Finds every op that consumes a tensor whose name contains a given substring, and prints the
// consumer chain a few hops deep, with any scalar Const/BinaryOp params along the way -- used to
// see what a graph does to an input tensor (e.g. does "timestep" get divided/scaled anywhere).
#include "MNN_generated.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <set>

using namespace MNN;

static void printOp(const Net* net, const Op* op) {
    printf("op name=%s type=%d (mainType=%d)\n", op->name() ? op->name()->c_str() : "", (int)op->type(), (int)op->main_type());
    if (op->main_type() == OpParameter_BinaryOp) {
        auto b = op->main_as_BinaryOp();
        printf("  BinaryOp opType=%d\n", (int)b->opType());
    } else if (op->main_type() == OpParameter_Blob) {
        auto blob = op->main_as_Blob();
        if (blob->dataType() == DataType_DT_FLOAT && blob->float32s()) {
            printf("  Blob float32s=[");
            for (int k = 0; k < (int)blob->float32s()->size() && k < 8; ++k) printf("%f,", blob->float32s()->Get(k));
            printf("]\n");
        }
    } else if (op->main_type() == OpParameter_Eltwise) {
        auto e = op->main_as_Eltwise();
        printf("  Eltwise type=%d\n", (int)e->type());
    }
    if (op->inputIndexes()) {
        for (int ii = 0; ii < (int)op->inputIndexes()->size(); ++ii) {
            int iidx = op->inputIndexes()->Get(ii);
            printf("  input[%d] idx=%d name=%s\n", ii, iidx, net->tensorName()->Get(iidx)->c_str());
        }
    }
    if (op->outputIndexes()) {
        for (int oo = 0; oo < (int)op->outputIndexes()->size(); ++oo) {
            int oidx = op->outputIndexes()->Get(oo);
            printf("  output[%d] idx=%d name=%s\n", oo, oidx, net->tensorName()->Get(oidx)->c_str());
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: dump_consumers <model.mnn> <tensor_name_substring> [hops=2]\n");
        return 1;
    }
    int hops = argc > 3 ? atoi(argv[3]) : 2;
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto net = GetNet(buf.data());

    std::set<int> frontier;
    for (int i = 0; i < (int)net->tensorName()->size(); ++i) {
        auto n = net->tensorName()->Get(i);
        if (n && std::string(n->c_str()).find(argv[2]) != std::string::npos) frontier.insert(i);
    }
    printf("seed tensors: %zu\n", frontier.size());
    for (int idx : frontier) printf("  idx=%d name=%s\n", idx, net->tensorName()->Get(idx)->c_str());

    std::set<int> visitedOps;
    for (int hop = 0; hop < hops; ++hop) {
        std::set<int> nextFrontier;
        printf("=== hop %d ===\n", hop);
        for (int i = 0; i < (int)net->oplists()->size(); ++i) {
            if (visitedOps.count(i)) continue;
            auto op = net->oplists()->Get(i);
            if (!op->inputIndexes()) continue;
            bool consumes = false;
            for (int ii = 0; ii < (int)op->inputIndexes()->size(); ++ii) {
                if (frontier.count(op->inputIndexes()->Get(ii))) consumes = true;
            }
            if (!consumes) continue;
            visitedOps.insert(i);
            printOp(net, op);
            if (op->outputIndexes()) {
                for (int oo = 0; oo < (int)op->outputIndexes()->size(); ++oo) nextFrontier.insert(op->outputIndexes()->Get(oo));
            }
        }
        frontier = nextFrontier;
    }
    return 0;
}
