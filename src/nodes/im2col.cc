#include "im2col.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include "../options.h"
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
    const Tensor* X = get_input_tensor(0);
    const Tensor* W = get_input_tensor(arithmetic_mode == QLinearConv ? 3 : 1);
    const Tensor* Y = get_output_tensor(0);
    int64_t in_ch_per_group = in_ch / group;
    ss << "FUNC_PREFIX void " << c_name() << "(const " << X->data_type_str() << " x[" << batch << "][" << in_ch << "][" << in_h << "][" << in_w << "], ";
    ss << "const " << W->data_type_str() << " w[" << out_ch << "][" << in_ch_per_group << "][" << kernel_h << "][" << kernel_w << "]";
    if (has_bias && arithmetic_mode == Conv) {
        const Tensor* B = get_input_tensor(2);
        ss << ", const " << B->data_type_str() << " bias[" << out_ch << "]";
    }
    ss << ", " << Y->data_type_str() << " y[" << batch << "][" << out_ch << "][" << out_h << "][" << out_w << "])";
    return ss.str();
}

void Im2Col::print_node(std::ostream& dst) const
{
    dst << "\t/* Fused Im2Col + MatMul (equivalent to conv but computed differently) */" << std::endl;

    dst << "\tfor(uint32_t b = 0; b < " << batch << "; b++) {" << std::endl;
    dst << "\t  for(uint32_t m = 0; m < " << out_ch << "; m++) {" << std::endl;
    dst << "\t    for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
    dst << "\t      for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
    print_accumulator_init(dst);
    dst << "\t        " << std::endl;

    if (group == 1) {
        dst << "\t        /* Standard convolution */" << std::endl;
        dst << "\t        for(uint32_t c = 0; c < " << in_ch << "; c++) {" << std::endl;
        dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "; ky++) {" << std::endl;
        dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "; kx++) {" << std::endl;

        if (stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 && pad_h == 0 && pad_w == 0) {
            dst << "\t              int ii0 = oy + ky;" << std::endl;
            dst << "\t              int ii1 = ox + kx;" << std::endl;
            print_accumulator_calc(dst, "x[b][c][ii0][ii1]", "w[m][c][ky][kx]");
        } else {
            dst << "\t              int ii0 = oy * " << stride_h << " + ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
            dst << "\t              int ii1 = ox * " << stride_w << " + kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
            dst << "\t              if(ii0 >= 0 && ii0 < " << in_h << " && ii1 >= 0 && ii1 < " << in_w << ")" << std::endl;
            dst << "\t              {" << std::endl;
            print_accumulator_calc(dst, "x[b][c][ii0][ii1]", "w[m][c][ky][kx]");
            dst << "\t              }" << std::endl;
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
        dst << "\t              {" << std::endl;
        print_accumulator_calc(dst, "x[b][c][ii0][ii1]", "w[m][c - c_start][ky][kx]");
        dst << "\t              }" << std::endl;
        dst << "\t            }" << std::endl;
        dst << "\t          }" << std::endl;
        dst << "\t        }" << std::endl;
    }

    dst << std::endl;
    print_accumulator_finalize(dst);
    dst << "\t      }" << std::endl;
    dst << "\t    }" << std::endl;
    dst << "\t  }" << std::endl;
    dst << "\t}" << std::endl;
}

void Im2Col::print_accumulator_init(std::ostream& dst) const
{
    if (arithmetic_mode == QLinearConv) {
        dst << "\t        int32_t acc = ";
        if (has_bias)
            dst << "bias[m]";
        else
            dst << "0";
        dst << ";" << std::endl;
        return;
    }

    const Tensor* Y = get_output_tensor(0);
    dst << "\t        " << Y->data_type_str() << " acc = ";
    if (arithmetic_mode == Conv && has_bias)
        dst << "bias[m]";
    else
        dst << "0";
    dst << ";" << std::endl;
}

void Im2Col::print_accumulator_calc(std::ostream& dst, const std::string& x_idx, const std::string& w_idx) const
{
    if (arithmetic_mode == Conv) {
        dst << "\t                acc += " << x_idx << " * " << w_idx << ";" << std::endl;
        return;
    }

    if (arithmetic_mode == ConvInteger) {
        std::string x_zero = has_x_zero_point ? "x_zero_point[0]" : "0";
        std::string w_zero = has_w_zero_point ? "w_zero_point[0]" : "0";
        dst << "\t                acc += (" << x_idx << " - " << x_zero << ") * (" << w_idx << " - " << w_zero << ");" << std::endl;
        return;
    }

    dst << "\t                acc += ((int32_t)" << x_idx << " - x_zero_point[0]) * ((int32_t)" << w_idx << " - w_zero_point[0]);" << std::endl;
}

void Im2Col::print_accumulator_finalize(std::ostream& dst) const
{
    if (arithmetic_mode == QLinearConv) {
        const Tensor* X_scale = get_input_tensor(1);
        const Tensor* Y = get_output_tensor(0);
        std::string float_dtype = X_scale->data_type_str();
        dst << "\t        " << float_dtype << " scaled = ((" << float_dtype << ")acc) * (x_scale[0] * w_scale[0]) / y_scale[0];" << std::endl;
        dst << "\t        scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
        dst << "\t        y[b][m][oy][ox] = (" << Y->data_type_str() << ") roundf(scaled);" << std::endl;
        return;
    }

    dst << "\t        y[b][m][oy][ox] = acc;" << std::endl;
}

void Im2Col::print_tiled_conv_node(std::ostream& dst) const
{
    const Tensor* X = get_input_tensor(0);
    const Tensor* Y = get_output_tensor(0);
    const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 16;
    const int64_t in_ch_per_group = in_ch / group;
    const int64_t out_ch_per_group = out_ch / group;
    const int64_t K = in_ch_per_group * kernel_h * kernel_w;

    dst << "\t/* Im2Col-lowered Conv: tiled matmul core */" << std::endl;
    dst << "\tconst uint32_t K = " << K << "u;" << std::endl;
    dst << "\tconst uint32_t MTILE = " << mtile << "u;" << std::endl;
    dst << "\t" << X->data_type_str() << " col[" << K << "];" << std::endl;
    dst << "\tfor(uint32_t b = 0; b < " << batch << "u; b++) {" << std::endl;
    dst << "\t  for(uint32_t g = 0; g < " << group << "u; g++) {" << std::endl;
    dst << "\t    for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
    dst << "\t      for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
    dst << "\t        uint32_t kk = 0;" << std::endl;
    dst << "\t        for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
    dst << "\t          uint32_t c = g * " << in_ch_per_group << "u + c0;" << std::endl;
    dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
    dst << "\t            int32_t iy = oy * " << stride_h << " + (int32_t)ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
    dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
    dst << "\t              int32_t ix = ox * " << stride_w << " + (int32_t)kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
    dst << "\t              col[kk++] = (iy >= 0 && iy < " << in_h << " && ix >= 0 && ix < " << in_w << ") ? x[b][c][iy][ix] : 0;" << std::endl;
    dst << "\t            }" << std::endl;
    dst << "\t          }" << std::endl;
    dst << "\t        }" << std::endl;
    dst << "\t        for(uint32_t m0 = g * " << out_ch_per_group << "u; m0 < (g + 1u) * " << out_ch_per_group << "u; m0 += MTILE) {" << std::endl;
    dst << "\t          uint32_t m1 = MIN(m0 + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
    dst << "\t          for(uint32_t m = m0; m < m1; m++) {" << std::endl;
    dst << "\t            " << Y->data_type_str() << " acc = " << (has_bias ? "bias[m]" : "0") << ";" << std::endl;
    dst << "\t            uint32_t k = 0;" << std::endl;
    dst << "\t            for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
    dst << "\t              for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
    dst << "\t                for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
    dst << "\t                  acc += col[k++] * w[m][c0][ky][kx];" << std::endl;
    dst << "\t                }" << std::endl;
    dst << "\t              }" << std::endl;
    dst << "\t            }" << std::endl;
    dst << "\t            y[b][m][oy][ox] = acc;" << std::endl;
    dst << "\t          }" << std::endl;
    dst << "\t        }" << std::endl;
    dst << "\t      }" << std::endl;
    dst << "\t    }" << std::endl;
    dst << "\t  }" << std::endl;
    dst << "\t}" << std::endl;
}

void Im2Col::print_tiled_quantized_conv_node(std::ostream& dst) const
{
    const bool qlinear = arithmetic_mode == QLinearConv;
    const Tensor* Y = get_output_tensor(0);
    const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 16;
    const int64_t in_ch_per_group = in_ch / group;
    const int64_t out_ch_per_group = out_ch / group;
    const int64_t K = in_ch_per_group * kernel_h * kernel_w;

    dst << "\t/* Im2Col-lowered " << (qlinear ? "QLinearConv" : "ConvInteger") << ": tiled integer matmul core */" << std::endl;
    dst << "\tconst uint32_t K = " << K << "u;" << std::endl;
    dst << "\tconst uint32_t MTILE = " << mtile << "u;" << std::endl;
    dst << "\tint32_t col[" << K << "];" << std::endl;
    dst << "\tint32_t x_zp = " << (qlinear || has_x_zero_point ? "(int32_t)x_zero_point[0]" : "0") << ";" << std::endl;
    dst << "\tint32_t w_zp = " << (qlinear || has_w_zero_point ? "(int32_t)w_zero_point[0]" : "0") << ";" << std::endl;
    dst << "\tfor(uint32_t b = 0; b < " << batch << "u; b++) {" << std::endl;
    dst << "\t  for(uint32_t g = 0; g < " << group << "u; g++) {" << std::endl;
    dst << "\t    for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
    dst << "\t      for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
    dst << "\t        uint32_t kk = 0;" << std::endl;
    dst << "\t        for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
    dst << "\t          uint32_t c = g * " << in_ch_per_group << "u + c0;" << std::endl;
    dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
    dst << "\t            int32_t iy = oy * " << stride_h << " + (int32_t)ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
    dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
    dst << "\t              int32_t ix = ox * " << stride_w << " + (int32_t)kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
    dst << "\t              col[kk++] = (iy >= 0 && iy < " << in_h << " && ix >= 0 && ix < " << in_w << ") ? ((int32_t)x[b][c][iy][ix] - x_zp) : 0;" << std::endl;
    dst << "\t            }" << std::endl;
    dst << "\t          }" << std::endl;
    dst << "\t        }" << std::endl;
    dst << "\t        for(uint32_t m0 = g * " << out_ch_per_group << "u; m0 < (g + 1u) * " << out_ch_per_group << "u; m0 += MTILE) {" << std::endl;
    dst << "\t          uint32_t m1 = MIN(m0 + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
    dst << "\t          for(uint32_t m = m0; m < m1; m++) {" << std::endl;
    dst << "\t            int32_t acc32 = " << (qlinear && has_bias ? "bias[m]" : "0") << ";" << std::endl;
    dst << "\t            uint32_t k = 0;" << std::endl;
    dst << "\t            for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
    dst << "\t              for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
    dst << "\t                for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
    dst << "\t                  acc32 += col[k++] * ((int32_t)w[m][c0][ky][kx] - w_zp);" << std::endl;
    dst << "\t                }" << std::endl;
    dst << "\t              }" << std::endl;
    dst << "\t            }" << std::endl;
    if (qlinear) {
        auto [lower, upper] = Y->get_type_bounds();
        std::string float_dtype = get_input_tensor(1)->data_type_str();
        dst << "\t            " << float_dtype << " scaled = ((" << float_dtype << ")acc32) * (x_scale[0] * w_scale[0]) / y_scale[0];" << std::endl;
        dst << "\t            scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
        dst << "\t            int32_t q = (int32_t)llround((double)scaled);" << std::endl;
        dst << "\t            q = MIN(MAX(q, " << lower << "), " << upper << ");" << std::endl;
        dst << "\t            y[b][m][oy][ox] = (" << Y->data_type_str() << ")q;" << std::endl;
    } else {
        dst << "\t            y[b][m][oy][ox] = acc32;" << std::endl;
    }
    dst << "\t          }" << std::endl;
    dst << "\t        }" << std::endl;
    dst << "\t      }" << std::endl;
    dst << "\t    }" << std::endl;
    dst << "\t  }" << std::endl;
    dst << "\t}" << std::endl;
}

void Im2Col::print_protected_conv_node(std::ostream& dst) const
{
    const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 16;
    const bool abyzft_enabled = options.abyzft_gemm;
    const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
    const bool freivalds_enabled = options.freivalds_gemm;
    const uint32_t randomized_checks = freivalds_enabled
        ? (options.freivalds_checks ? options.freivalds_checks : 1)
        : (options.gvfa_checks ? options.gvfa_checks : 1);
    const int64_t in_ch_per_group = in_ch / group;
    const int64_t out_ch_per_group = out_ch / group;
    const bool depthwise_single_output_per_group = group > 1 && out_ch_per_group == 1;
    const int64_t tiles_per_group = (out_ch_per_group + mtile - 1) / mtile;
    const int64_t K = in_ch_per_group * kernel_h * kernel_w;
    const std::string xtype = get_input_tensor(0)->data_type_str();
    const std::string ytype = get_output_tensor(0)->data_type_str();
    const Tensor* W = get_input_tensor(1);
    const Tensor* B_bias = has_bias ? get_input_tensor(2) : nullptr;
    const bool ct_checksums = !randomized_enabled
        && options.abft_weight_checksums_compiletime
        && W->isConst && W->data_buffer
        && W->data_type == onnx::TensorProto_DataType_FLOAT;
    const bool ct_bias = ct_checksums && has_bias && B_bias
        && B_bias->isConst && B_bias->data_buffer
        && B_bias->data_type == onnx::TensorProto_DataType_FLOAT;

    dst << "\t/* Im2Col-lowered Conv: protected matmul core */" << std::endl;
    dst << "\tconst uint32_t LAYER_ID = " << sweep_layer_id << ";" << std::endl;
    dst << "\tconst uint32_t K = " << K << "u;" << std::endl;
    dst << "\tconst uint32_t MTILE = " << mtile << "u;" << std::endl;
    dst << "\t" << xtype << " col[" << K << "];" << std::endl;
    if (ct_checksums) {
        float* wd = (float*)W->data_buffer;
        dst << "\tstatic const double b_rs_cache_ct[" << group << "][" << tiles_per_group << "][" << K << "] = {" << std::endl;
        for (int64_t cg = 0; cg < (int64_t)group; cg++) {
            dst << "\t  {" << std::endl;
            for (int64_t tile = 0; tile < tiles_per_group; tile++) {
                int64_t m0 = cg * out_ch_per_group + tile * mtile;
                int64_t m1 = std::min(m0 + (int64_t)mtile, (cg + 1) * out_ch_per_group);
                dst << "\t    {";
                int64_t k = 0;
                for (int64_t c0 = 0; c0 < in_ch_per_group; c0++) {
                    for (int64_t ky2 = 0; ky2 < kernel_h; ky2++) {
                        for (int64_t kx2 = 0; kx2 < kernel_w; kx2++) {
                            double s = 0.0;
                            for (int64_t m = m0; m < m1; m++) {
                                int idx = (int)((m * in_ch_per_group + c0) * kernel_h * kernel_w + ky2 * kernel_w + kx2);
                                s += (double)wd[idx];
                            }
                            if (k > 0) dst << ", ";
                            dst << s;
                            k++;
                        }
                    }
                }
                dst << "}";
                if (tile + 1 < tiles_per_group) dst << ",";
                dst << std::endl;
            }
            dst << "\t  }";
            if (cg + 1 < (int64_t)group) dst << ",";
            dst << std::endl;
        }
        dst << "\t};" << std::endl;
        if (ct_bias) {
            float* bd = (float*)B_bias->data_buffer;
            dst << "\tstatic const double bias_cache_ct[" << group << "][" << tiles_per_group << "] = {" << std::endl;
            for (int64_t cg = 0; cg < (int64_t)group; cg++) {
                dst << "\t  {";
                for (int64_t tile = 0; tile < tiles_per_group; tile++) {
                    int64_t m0 = cg * out_ch_per_group + tile * mtile;
                    int64_t m1 = std::min(m0 + (int64_t)mtile, (cg + 1) * out_ch_per_group);
                    double s = 0.0;
                    for (int64_t m = m0; m < m1; m++) s += (double)bd[m];
                    if (tile > 0) dst << ", ";
                    dst << s;
                }
                dst << "}";
                if (cg + 1 < (int64_t)group) dst << ",";
                dst << std::endl;
            }
            dst << "\t};" << std::endl;
        }
    }
    if (abyzft_enabled)
        dst << "\tstatic uint32_t __abyzft_n = 0; ++__abyzft_n;" << std::endl;
    dst << "\tfor(uint32_t b = 0; b < " << batch << "u; b++) {" << std::endl;
    dst << "\t  for(uint32_t g = 0; g < " << group << "u; g++) {" << std::endl;
    dst << "\t    const uint32_t TILES_PER_GROUP = " << tiles_per_group << "u;" << std::endl;
    if (!randomized_enabled) {
        dst << "\t    double abft_sumC_acc[" << tiles_per_group << "] = {0};" << std::endl;
        dst << "\t    double abft_pred_acc[" << tiles_per_group << "] = {0};" << std::endl;
    }
    if (randomized_enabled && depthwise_single_output_per_group) {
        dst << "\t    double randomized_sumC_acc[" << randomized_checks << "][" << tiles_per_group << "] = {{0}};" << std::endl;
        dst << "\t    double randomized_pred_acc[" << randomized_checks << "][" << tiles_per_group << "] = {{0}};" << std::endl;
    }

    // Pre-compute weight checksums ONCE per (b,g), NOT inside the spatial loop.
    // The original code recomputed b_rs for every output pixel (oy,ox), making
    // verification O(out_h*out_w*out_ch*K) — matching the matmul cost itself.
    // With caching it becomes O(out_ch*K) precompute + O(out_spatial*K) dot-products.
    if (randomized_enabled) {
        if (freivalds_enabled)
            dst << "\t    uint8_t r_cache[" << randomized_checks << "][" << tiles_per_group << "][" << mtile << "];" << std::endl;
        else
            dst << "\t    float r_cache[" << randomized_checks << "][" << tiles_per_group << "][" << mtile << "];" << std::endl;
        dst << "\t    double b_rs_cache[" << randomized_checks << "][" << tiles_per_group << "][" << K << "];" << std::endl;
        dst << "\t    double bias_cache[" << randomized_checks << "][" << tiles_per_group << "];" << std::endl;
        dst << "\t    for(uint32_t chk = 0; chk < " << randomized_checks << "u; chk++) {" << std::endl;
        dst << "\t      for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t        uint32_t m0_pre = g * " << out_ch_per_group << "u + tile * MTILE;" << std::endl;
        dst << "\t        uint32_t m1_pre = MIN(m0_pre + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
        dst << "\t        uint32_t rand_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ b ^ g ^ m0_pre ^ (uint32_t)(chk * 0x85EBCA6Bu));" << std::endl;
        if (freivalds_enabled) {
            dst << "\t        for(uint32_t mi = 0; mi < MTILE; mi++) {" << std::endl;
            dst << "\t          uint32_t active = (mi < m1_pre - m0_pre) ? ABYZFT_randbit(&rand_state) : 0u;" << std::endl;
            dst << "\t          r_cache[chk][tile][mi] = (uint8_t)active;" << std::endl;
            dst << "\t        }" << std::endl;
        } else {
            dst << "\t        for(uint32_t mi = 0; mi < MTILE; mi++) r_cache[chk][tile][mi] = (mi < m1_pre - m0_pre) ? ABYZFT_randn(&rand_state) : 0.0f;" << std::endl;
        }
        dst << "\t        bias_cache[chk][tile] = 0.0;" << std::endl;
        dst << "\t        for(uint32_t kk2 = 0; kk2 < K; kk2++) b_rs_cache[chk][tile][kk2] = 0.0;" << std::endl;
        dst << "\t        for(uint32_t m = m0_pre; m < m1_pre; m++) {" << std::endl;
        if (freivalds_enabled) {
            dst << "\t          if( !r_cache[chk][tile][m - m0_pre] ) continue;" << std::endl;
            if (has_bias) dst << "\t          bias_cache[chk][tile] += (double)bias[m];" << std::endl;
        } else {
            if (has_bias) dst << "\t          bias_cache[chk][tile] += (double)bias[m] * (double)r_cache[chk][tile][m - m0_pre];" << std::endl;
        }
        dst << "\t          uint32_t k = 0;" << std::endl;
        dst << "\t          for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
        if (freivalds_enabled)
            dst << "\t            b_rs_cache[chk][tile][k++] += (double)w[m][c0][ky][kx];" << std::endl;
        else
            dst << "\t            b_rs_cache[chk][tile][k++] += (double)w[m][c0][ky][kx] * (double)r_cache[chk][tile][m - m0_pre];" << std::endl;
        dst << "\t          }" << std::endl;
        dst << "\t        }" << std::endl;
        dst << "\t      }" << std::endl;
        dst << "\t    }" << std::endl;
    } else {
        // ABFT: precompute weight column sums (b_rs[k] = sum_m(w[m][k]))
        if (!ct_checksums)
            dst << "\t    double b_rs_cache[" << tiles_per_group << "][" << K << "];" << std::endl;
        if (!ct_bias)
            dst << "\t    double bias_cache[" << tiles_per_group << "];" << std::endl;
        dst << "\t    for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t      uint32_t m0_pre = g * " << out_ch_per_group << "u + tile * MTILE;" << std::endl;
        dst << "\t      uint32_t m1_pre = MIN(m0_pre + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
        if (!ct_bias)
            dst << "\t      bias_cache[tile] = 0.0;" << std::endl;
        if (!ct_checksums)
            dst << "\t      for(uint32_t kk2 = 0; kk2 < K; kk2++) b_rs_cache[tile][kk2] = 0.0;" << std::endl;
        dst << "\t      for(uint32_t m = m0_pre; m < m1_pre; m++) {" << std::endl;
        if (has_bias && !ct_bias) dst << "\t        bias_cache[tile] += (double)bias[m];" << std::endl;
        if (!ct_checksums) {
            dst << "\t        uint32_t k = 0;" << std::endl;
            dst << "\t        for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
            dst << "\t          b_rs_cache[tile][k++] += (double)w[m][c0][ky][kx];" << std::endl;
            dst << "\t        }" << std::endl;
        }
        dst << "\t      }" << std::endl;
        dst << "\t    }" << std::endl;
    }

    // AByzFT: precompute per-filter scale (one scaleB per output channel = per column of B)
    if (abyzft_enabled) {
        dst << "\t    float abyzft_scaleB_cache[" << tiles_per_group << "][" << mtile << "];" << std::endl;
        dst << "\t    for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t      uint32_t m0_pre = g * " << out_ch_per_group << "u + tile * MTILE;" << std::endl;
        dst << "\t      uint32_t m1_pre = MIN(m0_pre + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
        dst << "\t      uint32_t abyzft_b_state = (uint32_t)(0xA5A5A5A5u ^ LAYER_ID ^ __abyzft_n ^ b ^ g ^ m0_pre);" << std::endl;
        dst << "\t      for(uint32_t mi = 0; mi < MTILE; mi++)" << std::endl;
        dst << "\t        abyzft_scaleB_cache[tile][mi] = (mi < m1_pre - m0_pre) ? (0.25f + 3.75f * ABYZFT_rand01(&abyzft_b_state)) : 1.0f;" << std::endl;
        dst << "\t    }" << std::endl;
    }

    // Spatial loop: col gather, matmul, fault injection, then verify using cached checksums
    dst << "\t    for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
    dst << "\t      for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
    dst << "\t        uint32_t kk = 0;" << std::endl;
    dst << "\t        for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
    dst << "\t          uint32_t c = g * " << in_ch_per_group << "u + c0;" << std::endl;
    dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
    dst << "\t            int32_t iy = oy * " << stride_h << " + (int32_t)ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
    dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
    dst << "\t              int32_t ix = ox * " << stride_w << " + (int32_t)kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
    dst << "\t              col[kk++] = (iy >= 0 && iy < " << in_h << " && ix >= 0 && ix < " << in_w << ") ? x[b][c][iy][ix] : 0;" << std::endl;
    dst << "\t            }" << std::endl;
    dst << "\t          }" << std::endl;
    dst << "\t        }" << std::endl;
    // AByzFT: one scaleA per row of A = per spatial position, shared across all tiles
    if (abyzft_enabled) {
        dst << "\t        uint32_t abyzft_a_state = (uint32_t)(0xDEADBEEFu ^ LAYER_ID ^ b ^ g ^ (uint32_t)oy ^ (uint32_t)ox);" << std::endl;
        dst << "\t        float abyzft_scaleA = 0.25f + 3.75f * ABYZFT_rand01(&abyzft_a_state);" << std::endl;
        dst << "\t        float col_scaled[" << K << "];" << std::endl;
        dst << "\t        for(uint32_t kk2 = 0; kk2 < K; kk2++) col_scaled[kk2] = (float)col[kk2] * abyzft_scaleA;" << std::endl;
    }
    dst << "\t        for(uint32_t m0 = g * " << out_ch_per_group << "u; m0 < (g + 1u) * " << out_ch_per_group << "u; m0 += MTILE) {" << std::endl;
    dst << "\t          uint32_t m1 = MIN(m0 + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
    dst << "\t          uint32_t tile_idx = (m0 - g * " << out_ch_per_group << "u) / MTILE;" << std::endl;
    dst << "\t          " << ytype << " acc_tile[" << mtile << "];" << std::endl;
    dst << "\t          for(uint32_t m = m0; m < m1; m++) {" << std::endl;
    // Matmul: no bias in scaled part for AByzFT; non-AByzFT includes bias as usual
    if (abyzft_enabled) {
        dst << "\t            float acc = 0.0f;" << std::endl;
        dst << "\t            float acc_scaled = 0.0f;" << std::endl;
        dst << "\t            uint32_t k = 0;" << std::endl;
        dst << "\t            for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
        dst << "\t              for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
        dst << "\t                for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
        dst << "\t                  float w_scaled = (float)w[m][c0][ky][kx] * abyzft_scaleB_cache[tile_idx][m - m0];" << std::endl;
        dst << "\t                  acc_scaled += col_scaled[k++] * w_scaled;" << std::endl;
        dst << "\t                }" << std::endl;
        dst << "\t              }" << std::endl;
        dst << "\t            }" << std::endl;
        dst << "\t            float _sAB = abyzft_scaleA * abyzft_scaleB_cache[tile_idx][m - m0];" << std::endl;
        dst << "\t            acc = (_sAB != 0.0f) ? (acc_scaled / _sAB) : acc_scaled;" << std::endl;
    } else {
        dst << "\t            " << ytype << " acc = " << (has_bias ? "(float)bias[m]" : "0.0f") << ";" << std::endl;
        dst << "\t            uint32_t k = 0;" << std::endl;
        dst << "\t            for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
        dst << "\t              for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
        dst << "\t                for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
        dst << "\t                  acc += col[k++] * w[m][c0][ky][kx];" << std::endl;
        dst << "\t                }" << std::endl;
        dst << "\t              }" << std::endl;
        dst << "\t            }" << std::endl;
    }
    dst << "\t            if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
    dst << "\t              const uint32_t P = " << out_h << "u * " << out_w << "u;" << std::endl;
    dst << "\t              uint32_t out_idx = ((b * " << out_ch << "u + m) * P) + ((uint32_t)oy * " << out_w << "u + (uint32_t)ox);" << std::endl;
    dst << "\t              if( FAULT_MODEL==0 ) {" << std::endl;
    if (abyzft_enabled)
        dst << "\t                if( out_idx == FAULT_INDEX ) { acc_scaled += FAULT_VALUE; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
    else
        dst << "\t                if( out_idx == FAULT_INDEX ) { acc += FAULT_VALUE; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
    dst << "\t              } else if( FAULT_MODEL==1 ) {" << std::endl;
    dst << "\t                uint32_t base_m = (FAULT_INDEX / P) % " << out_ch << "u;" << std::endl;
    dst << "\t                uint32_t base_p = FAULT_INDEX % P;" << std::endl;
    dst << "\t                bool ok = (base_m + 1u < " << out_ch << "u) && (base_p + 1u < P);" << std::endl;
    dst << "\t                float delta = 0.0f;" << std::endl;
    if (depthwise_single_output_per_group) {
        dst << "\t                uint32_t base_row = base_p / " << out_w << "u;" << std::endl;
        dst << "\t                uint32_t base_col = base_p % " << out_w << "u;" << std::endl;
        dst << "\t                ok = (" << out_h << "u > 1u) && (" << out_w << "u > 1u);" << std::endl;
        dst << "\t                uint32_t row0 = MIN(base_row, " << out_h << "u - 2u);" << std::endl;
        dst << "\t                uint32_t col0 = MIN(base_col, " << out_w << "u - 2u);" << std::endl;
        dst << "\t                uint32_t b_off = FAULT_INDEX / (" << out_ch << "u * P);" << std::endl;
        dst << "\t                uint32_t base = ((b_off * " << out_ch << "u + base_m) * P) + (row0 * " << out_w << "u + col0);" << std::endl;
    }
    dst << "\t                if( ok ) {" << std::endl;
    if (depthwise_single_output_per_group) {
        dst << "\t                  if( out_idx == base ) delta = +FAULT_VALUE;" << std::endl;
        dst << "\t                  else if( out_idx == base + 1u ) delta = -FAULT_VALUE;" << std::endl;
        dst << "\t                  else if( out_idx == base + " << out_w << "u ) delta = -FAULT_VALUE;" << std::endl;
        dst << "\t                  else if( out_idx == base + " << out_w << "u + 1u ) delta = +FAULT_VALUE;" << std::endl;
    } else {
    dst << "\t                  if( out_idx == FAULT_INDEX ) delta = +FAULT_VALUE;" << std::endl;
    dst << "\t                  else if( out_idx == FAULT_INDEX + 1u ) delta = -FAULT_VALUE;" << std::endl;
    dst << "\t                  else if( out_idx == FAULT_INDEX + P ) delta = -FAULT_VALUE;" << std::endl;
    dst << "\t                  else if( out_idx == FAULT_INDEX + P + 1u ) delta = +FAULT_VALUE;" << std::endl;
    }
    dst << "\t                }" << std::endl;
    if (abyzft_enabled)
        dst << "\t                if( delta != 0.0f ) { acc_scaled += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
    else
        dst << "\t                if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
    dst << "\t              } else if( FAULT_MODEL==2 ) {" << std::endl;
    dst << "\t                float delta = 0.0f;" << std::endl;
    dst << "\t                for( uint32_t rr=0; rr<FAULT_N; rr++ ) {" << std::endl;
    dst << "\t                  uint32_t base = FAULT_INDEX + rr * FAULT_STRIDE;" << std::endl;
    dst << "\t                  uint32_t base_m = (base / P) % " << out_ch << "u;" << std::endl;
    dst << "\t                  uint32_t base_p = base % P;" << std::endl;
    dst << "\t                  bool ok = (base_m + 1u < " << out_ch << "u) && (base_p + 1u < P);" << std::endl;
    if (depthwise_single_output_per_group) {
        dst << "\t                  uint32_t base_row = base_p / " << out_w << "u;" << std::endl;
        dst << "\t                  uint32_t base_col = base_p % " << out_w << "u;" << std::endl;
        dst << "\t                  ok = (" << out_h << "u > 1u) && (" << out_w << "u > 1u);" << std::endl;
        dst << "\t                  uint32_t row0 = MIN(base_row, " << out_h << "u - 2u);" << std::endl;
        dst << "\t                  uint32_t col0 = MIN(base_col, " << out_w << "u - 2u);" << std::endl;
        dst << "\t                  uint32_t b_off = base / (" << out_ch << "u * P);" << std::endl;
        dst << "\t                  base = ((b_off * " << out_ch << "u + base_m) * P) + (row0 * " << out_w << "u + col0);" << std::endl;
    }
    dst << "\t                  if( !ok ) continue;" << std::endl;
    if (depthwise_single_output_per_group) {
        dst << "\t                  if( out_idx == base ) delta += +FAULT_VALUE;" << std::endl;
        dst << "\t                  else if( out_idx == base + 1u ) delta += -FAULT_VALUE;" << std::endl;
        dst << "\t                  else if( out_idx == base + " << out_w << "u ) delta += -FAULT_VALUE;" << std::endl;
        dst << "\t                  else if( out_idx == base + " << out_w << "u + 1u ) delta += +FAULT_VALUE;" << std::endl;
    } else {
    dst << "\t                  if( out_idx == base ) delta += +FAULT_VALUE;" << std::endl;
    dst << "\t                  else if( out_idx == base + 1u ) delta += -FAULT_VALUE;" << std::endl;
    dst << "\t                  else if( out_idx == base + P ) delta += -FAULT_VALUE;" << std::endl;
    dst << "\t                  else if( out_idx == base + P + 1u ) delta += +FAULT_VALUE;" << std::endl;
    }
    dst << "\t                }" << std::endl;
    if (abyzft_enabled)
        dst << "\t                if( delta != 0.0f ) { acc_scaled += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
    else
        dst << "\t                if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
    dst << "\t              }" << std::endl;
    dst << "\t            }" << std::endl;
    // AByzFT: descale then add bias
    if (abyzft_enabled) {
        dst << "\t            acc = (_sAB != 0.0f) ? acc_scaled / _sAB : acc_scaled;" << std::endl;
        if (has_bias) dst << "\t            acc += (float)bias[m];" << std::endl;
    }
    dst << "\t            y[b][m][oy][ox] = acc;" << std::endl;
    dst << "\t            acc_tile[m - m0] = acc;" << std::endl;
    dst << "\t          }" << std::endl;
    if (randomized_enabled) {
        dst << "\t          for(uint32_t chk = 0; chk < " << randomized_checks << "u; chk++) {" << std::endl;
        dst << "\t            double sumC = 0.0;" << std::endl;
        dst << "\t            for(uint32_t m = m0; m < m1; m++) {" << std::endl;
        if (freivalds_enabled) {
            dst << "\t              if( !r_cache[chk][tile_idx][m - m0] ) continue;" << std::endl;
            dst << "\t              sumC += (double)acc_tile[m - m0];" << std::endl;
        } else {
            dst << "\t              sumC += (double)acc_tile[m - m0] * (double)r_cache[chk][tile_idx][m - m0];" << std::endl;
        }
        dst << "\t            }" << std::endl;
        dst << "\t            double pred = bias_cache[chk][tile_idx];" << std::endl;
        dst << "\t            for(uint32_t kk2 = 0; kk2 < K; kk2++) pred += (double)col[kk2] * b_rs_cache[chk][tile_idx][kk2];" << std::endl;
        if (depthwise_single_output_per_group) {
            dst << "\t            randomized_sumC_acc[chk][tile_idx] += sumC;" << std::endl;
            dst << "\t            randomized_pred_acc[chk][tile_idx] += pred;" << std::endl;
        } else {
        dst << "\t            double diff = fabs(pred - sumC);" << std::endl;
        dst << "\t            double tol = " << options.abft_eps << " * (fabs(pred) + 1.0);" << std::endl;
        dst << "\t            if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
        }
        dst << "\t          }" << std::endl;
    } else {
        dst << "\t          double sumC = 0.0;" << std::endl;
        dst << "\t          for(uint32_t m = m0; m < m1; m++) sumC += (double)acc_tile[m - m0];" << std::endl;
        if (ct_bias)
            dst << "\t          double pred = bias_cache_ct[g][tile_idx];" << std::endl;
        else
            dst << "\t          double pred = bias_cache[tile_idx];" << std::endl;
        if (ct_checksums)
            dst << "\t          for(uint32_t kk2 = 0; kk2 < K; kk2++) pred += (double)col[kk2] * b_rs_cache_ct[g][tile_idx][kk2];" << std::endl;
        else
            dst << "\t          for(uint32_t kk2 = 0; kk2 < K; kk2++) pred += (double)col[kk2] * b_rs_cache[tile_idx][kk2];" << std::endl;
        dst << "\t          abft_sumC_acc[tile_idx] += sumC;" << std::endl;
        dst << "\t          abft_pred_acc[tile_idx] += pred;" << std::endl;
    }
    dst << "\t        }" << std::endl;
    dst << "\t      }" << std::endl;
    dst << "\t    }" << std::endl;
    if (!randomized_enabled) {
        dst << "\t    for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t      double pred_acc = abft_pred_acc[tile];" << std::endl;
        dst << "\t      double diff = fabs(pred_acc - abft_sumC_acc[tile]);" << std::endl;
        dst << "\t      double tol = " << (options.abyzft_gemm ? (options.abft_eps * 0.1f) : options.abft_eps) << " * (fabs(pred_acc) + 1.0);" << std::endl;
        dst << "\t      if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
        dst << "\t    }" << std::endl;
    }
    if (randomized_enabled && depthwise_single_output_per_group) {
        dst << "\t    for(uint32_t chk = 0; chk < " << randomized_checks << "u; chk++) {" << std::endl;
        dst << "\t      for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t        double pred_acc = randomized_pred_acc[chk][tile];" << std::endl;
        dst << "\t        double diff = fabs(pred_acc - randomized_sumC_acc[chk][tile]);" << std::endl;
        dst << "\t        double tol = " << options.abft_eps << " * (fabs(pred_acc) + 1.0);" << std::endl;
        dst << "\t        if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
        dst << "\t      }" << std::endl;
        dst << "\t    }" << std::endl;
    }
    dst << "\t  }" << std::endl;
    dst << "\t}" << std::endl;
}

void Im2Col::print_protected_quantized_conv_node(std::ostream& dst) const
{
    const bool qlinear = arithmetic_mode == QLinearConv;
    const Tensor* X = get_input_tensor(0);
    const Tensor* W = get_input_tensor(qlinear ? 3 : 1);
    const Tensor* Y = get_output_tensor(0);
    const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 16;
    const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
    const bool freivalds_enabled = options.freivalds_gemm;
    const uint32_t randomized_checks = freivalds_enabled
        ? (options.freivalds_checks ? options.freivalds_checks : 1)
        : (options.gvfa_checks ? options.gvfa_checks : 1);
    const int64_t in_ch_per_group = in_ch / group;
    const int64_t out_ch_per_group = out_ch / group;
    const bool depthwise_single_output_per_group = group > 1 && out_ch_per_group == 1;
    const int64_t tiles_per_group = (out_ch_per_group + mtile - 1) / mtile;
    const int64_t K = in_ch_per_group * kernel_h * kernel_w;
    const bool unsigned_quant = X->data_type == onnx::TensorProto_DataType_UINT8 ||
                                X->data_type == onnx::TensorProto_DataType_UINT16 ||
                                X->data_type == onnx::TensorProto_DataType_UINT32 ||
                                X->data_type == onnx::TensorProto_DataType_UINT64;
    const char* scale_picker = unsigned_quant ? "ABYZFT_pick_scale_u8" : "ABYZFT_pick_scale_s8";
    auto supports_i32_read = [](const Tensor* t) -> bool {
        switch (t->data_type) {
            case onnx::TensorProto_DataType_INT8:
            case onnx::TensorProto_DataType_UINT8:
            case onnx::TensorProto_DataType_INT16:
            case onnx::TensorProto_DataType_UINT16:
            case onnx::TensorProto_DataType_INT32:
            case onnx::TensorProto_DataType_UINT32:
            case onnx::TensorProto_DataType_INT64:
            case onnx::TensorProto_DataType_UINT64:
                return true;
            default:
                return false;
        }
    };
    auto get_tensor_elem_i32 = [](const Tensor* t, int idx) -> int32_t {
        switch (t->data_type) {
            case onnx::TensorProto_DataType_INT8: return static_cast<int32_t>(reinterpret_cast<int8_t*>(t->data_buffer)[idx]);
            case onnx::TensorProto_DataType_UINT8: return static_cast<int32_t>(reinterpret_cast<uint8_t*>(t->data_buffer)[idx]);
            case onnx::TensorProto_DataType_INT16: return static_cast<int32_t>(reinterpret_cast<int16_t*>(t->data_buffer)[idx]);
            case onnx::TensorProto_DataType_UINT16: return static_cast<int32_t>(reinterpret_cast<uint16_t*>(t->data_buffer)[idx]);
            case onnx::TensorProto_DataType_INT32: return reinterpret_cast<int32_t*>(t->data_buffer)[idx];
            case onnx::TensorProto_DataType_UINT32: return static_cast<int32_t>(reinterpret_cast<uint32_t*>(t->data_buffer)[idx]);
            case onnx::TensorProto_DataType_INT64: return static_cast<int32_t>(reinterpret_cast<int64_t*>(t->data_buffer)[idx]);
            case onnx::TensorProto_DataType_UINT64: return static_cast<int32_t>(reinterpret_cast<uint64_t*>(t->data_buffer)[idx]);
            default: return 0;
        }
    };
    const bool use_abyzft_i32_accum = !options.abyzft_wide_accumulator;
    bool use_compiletime_abyzft_wbase = false;
    int32_t w_zp_const = 0;
    if (options.abyzft_gemm && W->isConst && supports_i32_read(W)) {
        const Tensor* w_zp_t = nullptr;
        if (qlinear)
            w_zp_t = get_input_tensor(5);
        else if (has_w_zero_point)
            w_zp_t = get_input_tensor(3);
        if (w_zp_t == nullptr) {
            use_compiletime_abyzft_wbase = true;
            w_zp_const = 0;
        } else if (w_zp_t->isConst && supports_i32_read(w_zp_t)) {
            use_compiletime_abyzft_wbase = true;
            w_zp_const = get_tensor_elem_i32(w_zp_t, 0);
        }
    }
    bool ct_checksums = false;
    int32_t ct_w_zp_val = 0;
    if (!randomized_enabled && options.abft_weight_checksums_compiletime
        && W->isConst && W->data_buffer && supports_i32_read(W)) {
        const Tensor* ct_wz = qlinear ? get_input_tensor(5)
            : (has_w_zero_point ? get_input_tensor(3) : nullptr);
        if (ct_wz == nullptr) {
            ct_checksums = true;
        } else if (ct_wz->isConst && supports_i32_read(ct_wz)) {
            ct_checksums = true;
            ct_w_zp_val = get_tensor_elem_i32(ct_wz, 0);
        }
    }

    dst << "\t/* Im2Col-lowered " << (qlinear ? "QLinearConv" : "ConvInteger") << ": protected integer matmul core */" << std::endl;
    dst << "\tconst uint32_t LAYER_ID = " << sweep_layer_id << ";" << std::endl;
    dst << "\tconst uint32_t K = " << K << "u;" << std::endl;
    dst << "\tconst uint32_t MTILE = " << mtile << "u;" << std::endl;
    dst << "\tint32_t col[" << K << "];" << std::endl;
    dst << "\tint32_t x_zp = " << (qlinear || has_x_zero_point ? "(int32_t)x_zero_point[0]" : "0") << ";" << std::endl;
    dst << "\tint32_t w_zp = " << (qlinear || has_w_zero_point ? "(int32_t)w_zero_point[0]" : "0") << ";" << std::endl;
    if (ct_checksums) {
        dst << "\tstatic const int64_t b_rs_cache_ct[" << group << "][" << tiles_per_group << "][" << K << "] = {" << std::endl;
        for (int64_t cg = 0; cg < (int64_t)group; cg++) {
            dst << "\t  {" << std::endl;
            for (int64_t tile = 0; tile < tiles_per_group; tile++) {
                int64_t m0 = cg * out_ch_per_group + tile * mtile;
                int64_t m1 = std::min(m0 + (int64_t)mtile, (cg + 1) * out_ch_per_group);
                dst << "\t    {";
                int64_t k = 0;
                for (int64_t c0 = 0; c0 < in_ch_per_group; c0++) {
                    for (int64_t ky2 = 0; ky2 < kernel_h; ky2++) {
                        for (int64_t kx2 = 0; kx2 < kernel_w; kx2++) {
                            int64_t s = 0;
                            for (int64_t m = m0; m < m1; m++) {
                                int idx = (int)((m * in_ch_per_group + c0) * kernel_h * kernel_w + ky2 * kernel_w + kx2);
                                s += (int64_t)(get_tensor_elem_i32(W, idx) - ct_w_zp_val);
                            }
                            if (k > 0) dst << ", ";
                            dst << s;
                            k++;
                        }
                    }
                }
                dst << "}";
                if (tile + 1 < tiles_per_group) dst << ",";
                dst << std::endl;
            }
            dst << "\t  }";
            if (cg + 1 < (int64_t)group) dst << ",";
            dst << std::endl;
        }
        dst << "\t};" << std::endl;
    }
    if (options.abyzft_gemm)
        dst << "\tstatic uint32_t __abyzft_n = 0; ++__abyzft_n;" << std::endl;
    dst << "\tfor(uint32_t b = 0; b < " << batch << "u; b++) {" << std::endl;
    dst << "\t  for(uint32_t g = 0; g < " << group << "u; g++) {" << std::endl;
    dst << "\t    const uint32_t TILES_PER_GROUP = " << tiles_per_group << "u;" << std::endl;
    if (!randomized_enabled) {
        dst << "\t    int64_t abft_sumC_acc[" << tiles_per_group << "] = {0};" << std::endl;
        dst << "\t    int64_t abft_pred_acc[" << tiles_per_group << "] = {0};" << std::endl;
    }
    if (randomized_enabled && depthwise_single_output_per_group) {
        dst << "\t    double randomized_sumC_acc[" << randomized_checks << "][" << tiles_per_group << "] = {{0}};" << std::endl;
        dst << "\t    double randomized_pred_acc[" << randomized_checks << "][" << tiles_per_group << "] = {{0}};" << std::endl;
    }
    if (randomized_enabled) {
        if (freivalds_enabled) {
            dst << "\t    uint8_t r_cache[" << randomized_checks << "][" << tiles_per_group << "][" << mtile << "];" << std::endl;
        } else {
            dst << "\t    float r_cache[" << randomized_checks << "][" << tiles_per_group << "][" << mtile << "];" << std::endl;
        }
        dst << "\t    double b_rs_cache[" << randomized_checks << "][" << tiles_per_group << "][" << K << "];" << std::endl;
        dst << "\t    double bias_cache[" << randomized_checks << "][" << tiles_per_group << "];" << std::endl;
        dst << "\t    for(uint32_t chk = 0; chk < " << randomized_checks << "u; chk++) {" << std::endl;
        dst << "\t      for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t        uint32_t m0_pre = g * " << out_ch_per_group << "u + tile * MTILE;" << std::endl;
        dst << "\t        uint32_t m1_pre = MIN(m0_pre + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
        dst << "\t        uint32_t rand_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ b ^ g ^ m0_pre ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
        if (freivalds_enabled) {
            dst << "\t        for(uint32_t mi = 0; mi < MTILE; mi++) {" << std::endl;
            dst << "\t          uint32_t active = (mi < m1_pre - m0_pre) ? ABYZFT_randbit(&rand_state) : 0u;" << std::endl;
            dst << "\t          r_cache[chk][tile][mi] = (uint8_t)active;" << std::endl;
            dst << "\t        }" << std::endl;
        } else {
            dst << "\t        for(uint32_t mi = 0; mi < MTILE; mi++) r_cache[chk][tile][mi] = (mi < m1_pre - m0_pre) ? ABYZFT_randn(&rand_state) : 0.0f;" << std::endl;
        }
        dst << "\t        bias_cache[chk][tile] = 0.0;" << std::endl;
        dst << "\t        for(uint32_t kk2 = 0; kk2 < K; kk2++) b_rs_cache[chk][tile][kk2] = 0.0;" << std::endl;
        dst << "\t        for(uint32_t m = m0_pre; m < m1_pre; m++) {" << std::endl;
        if (freivalds_enabled) {
            dst << "\t          if( !r_cache[chk][tile][m - m0_pre] ) continue;" << std::endl;
            if (qlinear && has_bias) dst << "\t          bias_cache[chk][tile] += (double)bias[m];" << std::endl;
        } else {
            if (qlinear && has_bias) dst << "\t          bias_cache[chk][tile] += (double)bias[m] * (double)r_cache[chk][tile][m - m0_pre];" << std::endl;
        }
        dst << "\t          uint32_t k = 0;" << std::endl;
        dst << "\t          for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
        if (freivalds_enabled)
            dst << "\t            b_rs_cache[chk][tile][k++] += (double)((int32_t)w[m][c0][ky][kx] - w_zp);" << std::endl;
        else
            dst << "\t            b_rs_cache[chk][tile][k++] += (double)((int32_t)w[m][c0][ky][kx] - w_zp) * (double)r_cache[chk][tile][m - m0_pre];" << std::endl;
        dst << "\t          }" << std::endl;
        dst << "\t        }" << std::endl;
        dst << "\t      }" << std::endl;
        dst << "\t    }" << std::endl;
    } else {
        if (!ct_checksums)
            dst << "\t    int64_t b_rs_cache[" << tiles_per_group << "][" << K << "];" << std::endl;
        dst << "\t    int64_t bias_cache[" << tiles_per_group << "];" << std::endl;
        dst << "\t    for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t      uint32_t m0_pre = g * " << out_ch_per_group << "u + tile * MTILE;" << std::endl;
        dst << "\t      uint32_t m1_pre = MIN(m0_pre + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
        dst << "\t      bias_cache[tile] = 0;" << std::endl;
        if (!ct_checksums)
            dst << "\t      for(uint32_t kk2 = 0; kk2 < K; kk2++) b_rs_cache[tile][kk2] = 0;" << std::endl;
        dst << "\t      for(uint32_t m = m0_pre; m < m1_pre; m++) {" << std::endl;
        if (qlinear && has_bias) dst << "\t        bias_cache[tile] += (int64_t)bias[m];" << std::endl;
        if (!ct_checksums) {
            dst << "\t        uint32_t k = 0;" << std::endl;
            dst << "\t        for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
            dst << "\t          b_rs_cache[tile][k++] += (int64_t)((int32_t)w[m][c0][ky][kx] - w_zp);" << std::endl;
            dst << "\t        }" << std::endl;
        }
        dst << "\t      }" << std::endl;
        dst << "\t    }" << std::endl;
    }
    if (options.abyzft_gemm) {
        if (use_compiletime_abyzft_wbase) {
            dst << "\t    static const int16_t w_base_cache[" << out_ch << "][" << K << "] = {" << std::endl;
            for (int64_t m = 0; m < out_ch; m++) {
                dst << "\t      {";
                int64_t k = 0;
                for (int64_t c0 = 0; c0 < in_ch_per_group; c0++) {
                    for (int64_t ky = 0; ky < kernel_h; ky++) {
                        for (int64_t kx = 0; kx < kernel_w; kx++) {
                            if (k > 0)
                                dst << ", ";
                            const int idx = static_cast<int>((((m * in_ch_per_group) + c0) * kernel_h + ky) * kernel_w + kx);
                            dst << static_cast<int16_t>(get_tensor_elem_i32(W, idx) - w_zp_const);
                            k++;
                        }
                    }
                }
                dst << "}";
                if (m + 1 < out_ch)
                    dst << ",";
                dst << std::endl;
            }
            dst << "\t    };" << std::endl;
        }
        dst << "\t    int32_t abyzft_scaleB_cache[" << tiles_per_group << "][" << mtile << "];" << std::endl;
        dst << "\t    int16_t w_scaled_cache[" << tiles_per_group << "][" << mtile << "][" << K << "];" << std::endl;
        dst << "\t    for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t      uint32_t m0_pre = g * " << out_ch_per_group << "u + tile * MTILE;" << std::endl;
        dst << "\t      uint32_t m1_pre = MIN(m0_pre + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
        dst << "\t      uint32_t abyzft_b_state = (uint32_t)(0xA5A5A5A5u ^ LAYER_ID ^ __abyzft_n ^ b ^ g ^ m0_pre);" << std::endl;
        dst << "\t      for(uint32_t mi = 0; mi < MTILE; mi++) {" << std::endl;
        dst << "\t        abyzft_scaleB_cache[tile][mi] = (mi < m1_pre - m0_pre) ? (int32_t)" << scale_picker << "(&abyzft_b_state) : 1;" << std::endl;
        dst << "\t        for(uint32_t kk2 = 0; kk2 < K; kk2++) w_scaled_cache[tile][mi][kk2] = 0;" << std::endl;
        dst << "\t      }" << std::endl;
        dst << "\t      for(uint32_t m = m0_pre; m < m1_pre; m++) {" << std::endl;
        dst << "\t        uint32_t mi = m - m0_pre;" << std::endl;
        if (use_compiletime_abyzft_wbase) {
            dst << "\t        const int16_t* w_base = w_base_cache[m];" << std::endl;
            dst << "\t        for(uint32_t kk2 = 0; kk2 < K; kk2++) w_scaled_cache[tile][mi][kk2] = (int16_t)(w_base[kk2] * abyzft_scaleB_cache[tile][mi]);" << std::endl;
        } else {
            dst << "\t        uint32_t k = 0;" << std::endl;
            dst << "\t        for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
            dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
            dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
            dst << "\t              int32_t wq = ((int32_t)w[m][c0][ky][kx] - w_zp);" << std::endl;
            dst << "\t              w_scaled_cache[tile][mi][k++] = (int16_t)(wq * abyzft_scaleB_cache[tile][mi]);" << std::endl;
            dst << "\t            }" << std::endl;
            dst << "\t          }" << std::endl;
            dst << "\t        }" << std::endl;
        }
        dst << "\t      }" << std::endl;
        dst << "\t    }" << std::endl;
    }
    dst << "\t    for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
    dst << "\t      for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
    dst << "\t        uint32_t kk = 0;" << std::endl;
    dst << "\t        for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
    dst << "\t          uint32_t c = g * " << in_ch_per_group << "u + c0;" << std::endl;
    dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
    dst << "\t            int32_t iy = oy * " << stride_h << " + (int32_t)ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
    dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
    dst << "\t              int32_t ix = ox * " << stride_w << " + (int32_t)kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
    dst << "\t              col[kk++] = (iy >= 0 && iy < " << in_h << " && ix >= 0 && ix < " << in_w << ") ? ((int32_t)x[b][c][iy][ix] - x_zp) : 0;" << std::endl;
    dst << "\t            }" << std::endl;
    dst << "\t          }" << std::endl;
    dst << "\t        }" << std::endl;
    if (options.abyzft_gemm) {
        dst << "\t        uint32_t abyzft_a_state = (uint32_t)(0xDEADBEEFu ^ LAYER_ID ^ b ^ g ^ (uint32_t)oy ^ (uint32_t)ox);" << std::endl;
        dst << "\t        int32_t abyzft_scaleA = (int32_t)" << scale_picker << "(&abyzft_a_state);" << std::endl;
        dst << "\t        int16_t col_scaled[" << K << "];" << std::endl;
        dst << "\t        for(uint32_t kk2 = 0; kk2 < K; kk2++) col_scaled[kk2] = (int16_t)(col[kk2] * abyzft_scaleA);" << std::endl;
    }
    dst << "\t        for(uint32_t m0 = g * " << out_ch_per_group << "u; m0 < (g + 1u) * " << out_ch_per_group << "u; m0 += MTILE) {" << std::endl;
    dst << "\t          uint32_t m1 = MIN(m0 + MTILE, (g + 1u) * " << out_ch_per_group << "u);" << std::endl;
    dst << "\t          uint32_t tile_idx = (m0 - g * " << out_ch_per_group << "u) / MTILE;" << std::endl;
    dst << "\t          int32_t acc_tile[" << mtile << "];" << std::endl;
    dst << "\t          for(uint32_t m = m0; m < m1; m++) {" << std::endl;
    // Helper lambda to emit fault-injection body targeting a given accumulator variable
    auto emit_fault_inject = [&](const std::string& target) {
        dst << "\t            if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
        dst << "\t              const uint32_t P = " << out_h << "u * " << out_w << "u;" << std::endl;
        dst << "\t              uint32_t out_idx = ((b * " << out_ch << "u + m) * P) + ((uint32_t)oy * " << out_w << "u + (uint32_t)ox);" << std::endl;
        if (qlinear) {
            dst << "\t              double eff_scale = (double)x_scale[0] * (double)w_scale[0];" << std::endl;
            dst << "\t              int32_t fault_delta = eff_scale != 0.0 ? (int32_t)llround((double)FAULT_VALUE / eff_scale) : 0;" << std::endl;
        } else {
            dst << "\t              int32_t fault_delta = (int32_t)llround((double)FAULT_VALUE);" << std::endl;
        }
        dst << "\t              if( FAULT_MODEL==0 ) {" << std::endl;
        dst << "\t                if( out_idx == FAULT_INDEX ) { " << target << " += fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
        dst << "\t              } else if( FAULT_MODEL==1 ) {" << std::endl;
        dst << "\t                uint32_t base_m = (FAULT_INDEX / P) % " << out_ch << "u;" << std::endl;
        dst << "\t                uint32_t base_p = FAULT_INDEX % P;" << std::endl;
        dst << "\t                bool ok = (base_m + 1u < " << out_ch << "u) && (base_p + 1u < P);" << std::endl;
        dst << "\t                int32_t delta = 0;" << std::endl;
        if (depthwise_single_output_per_group) {
            dst << "\t                uint32_t base_row = base_p / " << out_w << "u;" << std::endl;
            dst << "\t                uint32_t base_col = base_p % " << out_w << "u;" << std::endl;
            dst << "\t                ok = (" << out_h << "u > 1u) && (" << out_w << "u > 1u);" << std::endl;
            dst << "\t                uint32_t row0 = MIN(base_row, " << out_h << "u - 2u);" << std::endl;
            dst << "\t                uint32_t col0 = MIN(base_col, " << out_w << "u - 2u);" << std::endl;
            dst << "\t                uint32_t b_off = FAULT_INDEX / (" << out_ch << "u * P);" << std::endl;
            dst << "\t                uint32_t base = ((b_off * " << out_ch << "u + base_m) * P) + (row0 * " << out_w << "u + col0);" << std::endl;
        }
        dst << "\t                if( ok ) {" << std::endl;
        if (depthwise_single_output_per_group) {
            dst << "\t                  if( out_idx == base ) delta = +fault_delta;" << std::endl;
            dst << "\t                  else if( out_idx == base + 1u ) delta = -fault_delta;" << std::endl;
            dst << "\t                  else if( out_idx == base + " << out_w << "u ) delta = -fault_delta;" << std::endl;
            dst << "\t                  else if( out_idx == base + " << out_w << "u + 1u ) delta = +fault_delta;" << std::endl;
        } else {
        dst << "\t                  if( out_idx == FAULT_INDEX ) delta = +fault_delta;" << std::endl;
        dst << "\t                  else if( out_idx == FAULT_INDEX + 1u ) delta = -fault_delta;" << std::endl;
        dst << "\t                  else if( out_idx == FAULT_INDEX + P ) delta = -fault_delta;" << std::endl;
        dst << "\t                  else if( out_idx == FAULT_INDEX + P + 1u ) delta = +fault_delta;" << std::endl;
        }
        dst << "\t                }" << std::endl;
        dst << "\t                if( delta != 0 ) { " << target << " += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
        dst << "\t              } else if( FAULT_MODEL==2 ) {" << std::endl;
        dst << "\t                int32_t delta = 0;" << std::endl;
        dst << "\t                for(uint32_t rr = 0; rr < FAULT_N; rr++) {" << std::endl;
        dst << "\t                  uint32_t base = FAULT_INDEX + rr * FAULT_STRIDE;" << std::endl;
        dst << "\t                  uint32_t base_m = (base / P) % " << out_ch << "u;" << std::endl;
        dst << "\t                  uint32_t base_p = base % P;" << std::endl;
        dst << "\t                  bool ok = (base_m + 1u < " << out_ch << "u) && (base_p + 1u < P);" << std::endl;
        if (depthwise_single_output_per_group) {
            dst << "\t                  uint32_t base_row = base_p / " << out_w << "u;" << std::endl;
            dst << "\t                  uint32_t base_col = base_p % " << out_w << "u;" << std::endl;
            dst << "\t                  ok = (" << out_h << "u > 1u) && (" << out_w << "u > 1u);" << std::endl;
            dst << "\t                  uint32_t row0 = MIN(base_row, " << out_h << "u - 2u);" << std::endl;
            dst << "\t                  uint32_t col0 = MIN(base_col, " << out_w << "u - 2u);" << std::endl;
            dst << "\t                  uint32_t b_off = base / (" << out_ch << "u * P);" << std::endl;
            dst << "\t                  base = ((b_off * " << out_ch << "u + base_m) * P) + (row0 * " << out_w << "u + col0);" << std::endl;
        }
        dst << "\t                  if( !ok ) continue;" << std::endl;
        if (depthwise_single_output_per_group) {
            dst << "\t                  if( out_idx == base ) delta += +fault_delta;" << std::endl;
            dst << "\t                  else if( out_idx == base + 1u ) delta += -fault_delta;" << std::endl;
            dst << "\t                  else if( out_idx == base + " << out_w << "u ) delta += -fault_delta;" << std::endl;
            dst << "\t                  else if( out_idx == base + " << out_w << "u + 1u ) delta += +fault_delta;" << std::endl;
        } else {
        dst << "\t                  if( out_idx == base ) delta += +fault_delta;" << std::endl;
        dst << "\t                  else if( out_idx == base + 1u ) delta += -fault_delta;" << std::endl;
        dst << "\t                  else if( out_idx == base + P ) delta += -fault_delta;" << std::endl;
        dst << "\t                  else if( out_idx == base + P + 1u ) delta += +fault_delta;" << std::endl;
        }
        dst << "\t                }" << std::endl;
        dst << "\t                if( delta != 0 ) { " << target << " += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
        dst << "\t              }" << std::endl;
        dst << "\t            }" << std::endl;
    };
    if (options.abyzft_gemm) {
        dst << "\t            int32_t acc32 = " << (qlinear && has_bias ? "bias[m]" : "0") << ";" << std::endl;
        dst << "\t            " << (use_abyzft_i32_accum ? "int32_t" : "int64_t") << " acc_scaled = 0;" << std::endl;
        dst << "\t            const int16_t* w_scaled = w_scaled_cache[tile_idx][m - m0];" << std::endl;
        if (use_abyzft_i32_accum)
            dst << "\t            for(uint32_t kk2 = 0; kk2 < K; kk2++) acc_scaled += (int32_t)col_scaled[kk2] * (int32_t)w_scaled[kk2];" << std::endl;
        else
            dst << "\t            for(uint32_t kk2 = 0; kk2 < K; kk2++) acc_scaled += (int64_t)col_scaled[kk2] * (int64_t)w_scaled[kk2];" << std::endl;
        dst << "\t            int64_t scaleAB = (int64_t)abyzft_scaleA * (int64_t)abyzft_scaleB_cache[tile_idx][m - m0];" << std::endl;
        dst << "\t            int64_t scaleAB_mag = (scaleAB < 0) ? -scaleAB : scaleAB;" << std::endl;
        dst << "\t            int32_t scaleAB_sign = (scaleAB < 0) ? -1 : 1;" << std::endl;
        dst << "\t            uint32_t scaleAB_shift = ABYZFT_pow2_shift_u64((uint64_t)scaleAB_mag);" << std::endl;
        // Fault injected into acc_scaled BEFORE descaling: Byzantine pair (+d,-d) in acc_scaled
        // is descaled asymmetrically when scaleB[m]!=scaleB[m+1], so plain sum catches it.
        emit_fault_inject("acc_scaled");
        dst << "\t            if( scaleAB_mag != 0 ) acc32 += (int32_t)(scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled, scaleAB_shift));" << std::endl;
    } else {
        dst << "\t            int32_t acc32 = " << (qlinear && has_bias ? "bias[m]" : "0") << ";" << std::endl;
        dst << "\t            uint32_t k = 0;" << std::endl;
        dst << "\t            for(uint32_t c0 = 0; c0 < " << in_ch_per_group << "u; c0++) {" << std::endl;
        dst << "\t              for(uint32_t ky = 0; ky < " << kernel_h << "u; ky++) {" << std::endl;
        dst << "\t                for(uint32_t kx = 0; kx < " << kernel_w << "u; kx++) {" << std::endl;
        dst << "\t                  acc32 += col[k++] * ((int32_t)w[m][c0][ky][kx] - w_zp);" << std::endl;
        dst << "\t                }" << std::endl;
        dst << "\t              }" << std::endl;
        dst << "\t            }" << std::endl;
        emit_fault_inject("acc32");
    }
    dst << "\t            acc_tile[m - m0] = acc32;" << std::endl;
    if (qlinear) {
        auto [lower, upper] = Y->get_type_bounds();
        std::string float_dtype = get_input_tensor(1)->data_type_str();
        dst << "\t            " << float_dtype << " scaled = ((" << float_dtype << ")acc32) * (x_scale[0] * w_scale[0]) / y_scale[0];" << std::endl;
        dst << "\t            scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
        dst << "\t            int32_t q = (int32_t)llround((double)scaled);" << std::endl;
        dst << "\t            q = MIN(MAX(q, " << lower << "), " << upper << ");" << std::endl;
        dst << "\t            y[b][m][oy][ox] = (" << Y->data_type_str() << ")q;" << std::endl;
    } else {
        dst << "\t            y[b][m][oy][ox] = acc32;" << std::endl;
    }
    dst << "\t          }" << std::endl;
    if (randomized_enabled) {
        dst << "\t          for(uint32_t chk = 0; chk < " << randomized_checks << "u; chk++) {" << std::endl;
        dst << "\t            double sumC = 0.0;" << std::endl;
        dst << "\t            for(uint32_t m = m0; m < m1; m++) {" << std::endl;
        if (freivalds_enabled) {
            dst << "\t              if( !r_cache[chk][tile_idx][m - m0] ) continue;" << std::endl;
            dst << "\t              sumC += (double)acc_tile[m - m0];" << std::endl;
        } else {
            dst << "\t              sumC += (double)acc_tile[m - m0] * (double)r_cache[chk][tile_idx][m - m0];" << std::endl;
        }
        dst << "\t            }" << std::endl;
        dst << "\t            double pred = bias_cache[chk][tile_idx];" << std::endl;
        dst << "\t            for(uint32_t kk2 = 0; kk2 < K; kk2++) pred += (double)col[kk2] * b_rs_cache[chk][tile_idx][kk2];" << std::endl;
        if (depthwise_single_output_per_group) {
            dst << "\t            randomized_sumC_acc[chk][tile_idx] += sumC;" << std::endl;
            dst << "\t            randomized_pred_acc[chk][tile_idx] += pred;" << std::endl;
        } else {
        dst << "\t            double diff = fabs(pred - sumC);" << std::endl;
        dst << "\t            double tol = " << options.abft_eps << " * (fabs(pred) + 1.0);" << std::endl;
        dst << "\t            if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
        }
        dst << "\t          }" << std::endl;
    } else {
        dst << "\t          int64_t sumC = 0;" << std::endl;
        dst << "\t          for(uint32_t m = m0; m < m1; m++) {" << std::endl;
        dst << "\t            sumC += (int64_t)acc_tile[m - m0];" << std::endl;
        dst << "\t          }" << std::endl;
        dst << "\t          int64_t pred = bias_cache[tile_idx];" << std::endl;
        if (ct_checksums)
            dst << "\t          for(uint32_t kk2 = 0; kk2 < K; kk2++) pred += (int64_t)col[kk2] * b_rs_cache_ct[g][tile_idx][kk2];" << std::endl;
        else
            dst << "\t          for(uint32_t kk2 = 0; kk2 < K; kk2++) pred += (int64_t)col[kk2] * b_rs_cache[tile_idx][kk2];" << std::endl;
        dst << "\t          abft_sumC_acc[tile_idx] += sumC;" << std::endl;
        dst << "\t          abft_pred_acc[tile_idx] += pred;" << std::endl;
    }
    dst << "\t        }" << std::endl;
    dst << "\t      }" << std::endl;
    dst << "\t    }" << std::endl;
    if (!randomized_enabled) {
        dst << "\t    for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t      if( abft_sumC_acc[tile] != abft_pred_acc[tile] ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
        dst << "\t    }" << std::endl;
    }
    if (randomized_enabled && depthwise_single_output_per_group) {
        dst << "\t    for(uint32_t chk = 0; chk < " << randomized_checks << "u; chk++) {" << std::endl;
        dst << "\t      for(uint32_t tile = 0; tile < TILES_PER_GROUP; tile++) {" << std::endl;
        dst << "\t        double pred_acc = randomized_pred_acc[chk][tile];" << std::endl;
        dst << "\t        double diff = fabs(pred_acc - randomized_sumC_acc[chk][tile]);" << std::endl;
        dst << "\t        double tol = " << options.abft_eps << " * (fabs(pred_acc) + 1.0);" << std::endl;
        dst << "\t        if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
        dst << "\t      }" << std::endl;
        dst << "\t    }" << std::endl;
    }
    dst << "\t  }" << std::endl;
    dst << "\t}" << std::endl;
    (void)W;
}

void Im2Col::resolve_datatypes(void)
{
}

void Im2Col::print(std::ostream& dst) const
{
    const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || options.freivalds_gemm || options.gvfa_gemm;
    if (arithmetic_mode == Conv) {
        if (checksum_enabled)
            print_protected_conv_node(dst);
        else
            print_tiled_conv_node(dst);
        return;
    }
    if (arithmetic_mode == ConvInteger || arithmetic_mode == QLinearConv) {
        if (checksum_enabled)
            print_protected_quantized_conv_node(dst);
        else
            print_tiled_quantized_conv_node(dst);
        return;
    }
    print_node(dst);
}

}
