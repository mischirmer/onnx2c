#include "im2col.h"

#include <iostream>
#include <sstream>
#include "../tensor.h"

namespace toC {

Im2Col::Im2Col()
{
    op_name = "Im2Col";
}

std::string Im2Col::c_name(void) const
{
    return "node_" + onnx_name;
}

std::string Im2Col::c_output_decl(void)
{
    std::stringstream ss;
    ss << "FUNC_PREFIX void " << c_name() << "(const float x[" << batch << "][" << in_ch << "][" << in_h << "][" << in_w << "], ";
    ss << "const float w[" << out_ch << "][" << in_ch << "][" << kernel_h << "][" << kernel_w << "]";
    if (has_bias) {
        ss << ", const float b[" << out_ch << "]";
    }
    ss << ", float y[" << batch << "][" << out_ch << "][" << out_h << "][" << out_w << "])";
    return ss.str();
}

void Im2Col::print_node(std::ostream& dst) const
{
    dst << "\t/* Fused Im2Col + MatMul (equivalent to conv but computed differently) */" << std::endl;
    
    dst << "\tfor(uint32_t b = 0; b < " << batch << "; b++) {" << std::endl;
    dst << "\t  for(uint32_t m = 0; m < " << out_ch << "; m++) {" << std::endl;
    dst << "\t    for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
    dst << "\t      for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
    dst << "\t        float acc = 0.0f;" << std::endl;
    dst << "\t        " << std::endl;

    if (group == 1) {
        dst << "\t        /* Standard convolution */" << std::endl;
        dst << "\t        for(uint32_t c = 0; c < " << in_ch << "; c++) {" << std::endl;
        dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "; ky++) {" << std::endl;
        dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "; kx++) {" << std::endl;

        if (stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 && pad_h == 0 && pad_w == 0) {
            dst << "\t              int ii0 = oy + ky;" << std::endl;
            dst << "\t              int ii1 = ox + kx;" << std::endl;
            dst << "\t              acc += x[b][c][ii0][ii1] * w[m][c][ky][kx];" << std::endl;
        } else {
            dst << "\t              int ii0 = oy * " << stride_h << " + ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
            dst << "\t              int ii1 = ox * " << stride_w << " + kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
            dst << "\t              if(ii0 >= 0 && ii0 < " << in_h << " && ii1 >= 0 && ii1 < " << in_w << ")" << std::endl;
            dst << "\t                acc += x[b][c][ii0][ii1] * w[m][c][ky][kx];" << std::endl;
        }

        dst << "\t            }" << std::endl;
        dst << "\t          }" << std::endl;
        dst << "\t        }" << std::endl;
    } else {
        uint32_t goc = out_ch / group;
        uint32_t gic = in_ch / group;
        dst << "\t        /* Grouped convolution */" << std::endl;
        dst << "\t        uint32_t g = m / " << goc << ";" << std::endl;
        dst << "\t        uint32_t c_start = g * " << gic << ";" << std::endl;
        dst << "\t        for(uint32_t c = c_start; c < c_start + " << gic << "; c++) {" << std::endl;
        dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "; ky++) {" << std::endl;
        dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "; kx++) {" << std::endl;
        dst << "\t              int ii0 = oy * " << stride_h << " + ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
        dst << "\t              int ii1 = ox * " << stride_w << " + kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
        dst << "\t              if(ii0 >= 0 && ii0 < " << in_h << " && ii1 >= 0 && ii1 < " << in_w << ")" << std::endl;
        dst << "\t                acc += x[b][c][ii0][ii1] * w[m][c - c_start][ky][kx];" << std::endl;
        dst << "\t            }" << std::endl;
        dst << "\t          }" << std::endl;
        dst << "\t        }" << std::endl;
    }

    if (has_bias) {
        dst << std::endl;
        dst << "\t        acc += b[m];" << std::endl;
    }

    dst << std::endl;
    dst << "\t        y[b][m][oy][ox] = acc;" << std::endl;
    dst << "\t      }" << std::endl;
    dst << "\t    }" << std::endl;
    dst << "\t  }" << std::endl;
    dst << "\t}" << std::endl;
}

void Im2Col::resolve_datatypes(void)
{
}

void Im2Col::print(std::ostream& dst) const
{
    print_node(dst);
}

}