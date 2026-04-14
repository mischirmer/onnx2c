/* This file is part of onnx2c.
 *
 * ConvInteger
 * Calculates an integer version of the convolution filter.
 *
 * Compared to the default Conv, the ConvInteger inputs (data
 * and weights both) are quantized with an offset. Presumably
 * this is to give better dynamic range for variables not centered
 * around zero.
 * These zero-point offsets are given as optional input tensors.
 */
#include "spatialfilter.h"
#include "../options.h"
namespace toC {

class ConvInteger : public SpatialFilter {
	public:
	ConvInteger()
	{
		op_name = "ConvInteger";
		auto_pad = "NOTSET";
		group = 1;
	}

	virtual void print_output_cell_init(std::ostream& dst, const std::string& y_idx) const override
	{
		INDT_3 << "y" << y_idx << " = 0;" << std::endl;
	}

	virtual void print_output_cell_calc(
	    std::ostream& dst,
	    const std::string& x_idx,
	    const std::string& w_idx,
	    const std::string& y_idx) const override
	{
		std::string x_zero = "0";
		if (get_number_of_inputs() >= 3) // x_zero_point is optional, 3rd input
			x_zero = constant_acces_code("x_zero_point[0]");

		std::string w_zero = "0";
		if (get_number_of_inputs() >= 4) // w_zero_point is optional, 4th input
			w_zero = constant_acces_code("w_zero_point[0]");

		INDT_4 << "y" << y_idx << " += (x " << x_idx << "  - " << x_zero << ") * (w" << w_idx << " -" << w_zero << ");" << std::endl;
	}

	virtual void print_output_cell_finalize(std::ostream& dst, const std::string& y_idx) const override
	{
	}

	virtual void print(std::ostream& dst) const override
	{
		print_header_info_comment(dst);
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || options.freivalds_gemm;
		if (options.conv_im2col || checksum_enabled) {
			print_im2col_matmul(dst);
		}
		else {
			print_loop_with_padding_checks(dst);
		}
	}

	virtual void resolve(void) override
	{
		name_input(0, "x");
		name_input(1, "w");

		if (get_number_of_inputs() > 2)
			name_input(2, "x_zero_point");
		if (get_number_of_inputs() > 3) {
			name_input(3, "w_zero_point");
		}

		if (get_X()->data_dim.size() != 4)
			ERROR("Unimplemented: ConvInteger for non 2D images");

		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		if (group != 1)
			ERROR("Unimplemented: ConvInteger: setting group to anything but 1");

		for (int d : dilations)
			if (d != 1)
				ERROR("Unimplemented: ConvInteger: dilations other than 1");

		Tensor* rv = new Tensor;
		rv->data_dim = resolve_output_size();
		rv->data_type = onnx::TensorProto_DataType_INT32;
		register_output(rv, "y");
	}

	void print_im2col_matmul(std::ostream& dst) const
	{
		const auto* x = get_X();
		const auto* y = get_Y();
		const uint32_t n_data_dims = get_numDataDim();

		if (n_data_dims != 2) {
			LOG(WARNING) << "ConvInteger im2col backend only supports 2D conv; falling back to direct loops for " << onnx_name << std::endl;
			print_loop_with_padding_checks(dst);
			return;
		}

		const uint32_t batch_size = x->data_dim[0];
		const uint32_t channels = x->data_dim[1];
		const uint32_t maps = y->data_dim[1];

		const int32_t in0 = x->data_dim[2];
		const int32_t in1 = x->data_dim[3];
		const int32_t out0 = y->data_dim[2];
		const int32_t out1 = y->data_dim[3];
		const int32_t k0 = kernel_shape[0];
		const int32_t k1 = kernel_shape[1];
		const int32_t s0 = strides[0];
		const int32_t s1 = strides[1];
		const int32_t d0 = dilations[0];
		const int32_t d1 = dilations[1];
		const int32_t pad0 = pads[0];
		const int32_t pad1 = pads[1];

		const uint32_t go = (group > 0) ? (maps / group) : maps;
		const uint32_t gi = (group > 0) ? (channels / group) : channels;
		const uint32_t K = gi * k0 * k1;

		INDT_1 << "/* ConvInteger (im2col + dot-product) */" << std::endl;
		INDT_1 << "const uint32_t LAYER_ID = " << node_id << ";" << std::endl;
		INDT_1 << "const uint32_t K = " << K << ";" << std::endl;
		INDT_1 << "int32_t col[" << K << "];" << std::endl;

		INDT_1 << "int32_t x_zp = 0;" << std::endl;
		if (get_number_of_inputs() >= 3)
			INDT_1 << "x_zp = (int32_t)x_zero_point[0];" << std::endl;
		INDT_1 << "int32_t w_zp = 0;" << std::endl;
		if (get_number_of_inputs() >= 4)
			INDT_1 << "w_zp = (int32_t)w_zero_point[0];" << std::endl;

		INDT_1 << "for( uint32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		if (group > 1) {
			INDT_1 << "uint32_t go = " << go << "; // output group size, i.e. maps/group" << std::endl;
			INDT_1 << "uint32_t gi = " << gi << "; // input group size, i.e. channels/group" << std::endl;
			INDT_1 << "for( uint32_t g=0; g<" << group << "; g++ ) {" << std::endl;
		}

		const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 16;
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || options.freivalds_gemm;
		const uint32_t freivalds_checks = options.freivalds_checks ? options.freivalds_checks : 1;

		std::string m_begin = (group > 1) ? "go*g" : "0";
		std::string m_end = (group > 1) ? "go*(g+1)" : std::to_string(maps);
		const uint32_t tiles_per_group = ((group > 1 ? go : maps) + mtile - 1) / mtile;

		INDT_2 << "const uint32_t MTILE = " << mtile << ";" << std::endl;
		INDT_2 << "const uint32_t m_base = (uint32_t)(" << m_begin << ");" << std::endl;
		INDT_2 << "const uint32_t m_limit = (uint32_t)(" << m_end << ");" << std::endl;
		if (checksum_enabled && options.freivalds_gemm) {
			INDT_2 << "/* Precompute Freivalds weighted B-reductions once per tile/check */" << std::endl;
			INDT_2 << "uint8_t freivalds_r_cache[" << freivalds_checks << "][" << tiles_per_group << "][" << mtile << "];" << std::endl;
			INDT_2 << "int32_t freivalds_brs_cache[" << freivalds_checks << "][" << tiles_per_group << "][" << K << "];" << std::endl;
			INDT_2 << "for( uint32_t chk=0; chk<" << freivalds_checks << "; chk++ ) {" << std::endl;
			INDT_3 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_4 << "uint32_t m0_pre = m_base + t*MTILE;" << std::endl;
			INDT_4 << "if( m0_pre >= m_limit ) continue;" << std::endl;
			INDT_4 << "uint32_t m1_pre = MIN(m0_pre + MTILE, m_limit);" << std::endl;
			INDT_4 << "uint32_t freivalds_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ (uint32_t)b ^ (uint32_t)m0_pre ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
			INDT_4 << "uint32_t r_any = 0;" << std::endl;
			INDT_4 << "for( uint32_t mi=0; mi<(m1_pre-m0_pre); mi++ ) { uint32_t bit = ABYZFT_randbit(&freivalds_state); freivalds_r_cache[chk][t][mi] = (uint8_t)bit; r_any |= bit; }" << std::endl;
			INDT_4 << "if( !r_any && (m1_pre>m0_pre) ) freivalds_r_cache[chk][t][0] = 1u;" << std::endl;
			INDT_4 << "for( uint32_t kk2=0; kk2<K; kk2++ ) freivalds_brs_cache[chk][t][kk2] = 0;" << std::endl;
			INDT_4 << "for( uint32_t m=m0_pre; m<m1_pre; m++ ) {" << std::endl;
			INDT_5 << "int32_t r = (int32_t)freivalds_r_cache[chk][t][m-m0_pre];" << std::endl;
			INDT_5 << "if( r == 0 ) continue;" << std::endl;
			INDT_5 << "uint32_t kk2 = 0;" << std::endl;
			INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
			INDT_5 << "freivalds_brs_cache[chk][t][kk2++] += (((int32_t)w[m][c0][kk0][kk1]) - w_zp);" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_4 << "}" << std::endl;
			INDT_3 << "}" << std::endl;
			INDT_2 << "}" << std::endl;
		}
		if (checksum_enabled && !options.freivalds_gemm) {
			auto supports_compiletime_i32_read = [](const Tensor* t) -> bool {
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

			auto get_tensor_elem_i32 = [](const Tensor* t, uint64_t idx) -> int32_t {
				switch (t->data_type) {
					case onnx::TensorProto_DataType_INT8:
						return static_cast<int32_t>(reinterpret_cast<int8_t*>(t->data_buffer)[idx]);
					case onnx::TensorProto_DataType_UINT8:
						return static_cast<int32_t>(reinterpret_cast<uint8_t*>(t->data_buffer)[idx]);
					case onnx::TensorProto_DataType_INT16:
						return static_cast<int32_t>(reinterpret_cast<int16_t*>(t->data_buffer)[idx]);
					case onnx::TensorProto_DataType_UINT16:
						return static_cast<int32_t>(reinterpret_cast<uint16_t*>(t->data_buffer)[idx]);
					case onnx::TensorProto_DataType_INT32:
						return reinterpret_cast<int32_t*>(t->data_buffer)[idx];
					case onnx::TensorProto_DataType_UINT32:
						return static_cast<int32_t>(reinterpret_cast<uint32_t*>(t->data_buffer)[idx]);
					case onnx::TensorProto_DataType_INT64:
						return static_cast<int32_t>(reinterpret_cast<int64_t*>(t->data_buffer)[idx]);
					case onnx::TensorProto_DataType_UINT64:
						return static_cast<int32_t>(reinterpret_cast<uint64_t*>(t->data_buffer)[idx]);
					default:
						return 0;
				}
			};

			bool use_compiletime_weight_checksums = false;
			int32_t w_zp_const = 0;
			if (options.abft_weight_checksums_compiletime && group == 1 && get_W()->isConst) {
				use_compiletime_weight_checksums = supports_compiletime_i32_read(get_W());
				if (get_number_of_inputs() >= 4) {
					const Tensor* w_zp_t = get_input_tensor(3);
					if (!w_zp_t->isConst || !supports_compiletime_i32_read(w_zp_t)) {
						use_compiletime_weight_checksums = false;
					}
					else {
						w_zp_const = get_tensor_elem_i32(w_zp_t, 0);
					}
				}
			}

			if (use_compiletime_weight_checksums) {
				INDT_2 << "/* Precomputed B tile checksums (compile-time) */" << std::endl;
				INDT_2 << "static const int32_t b_rs_cache[" << tiles_per_group << "][" << K << "] = {" << std::endl;
				for (uint32_t t = 0; t < tiles_per_group; t++) {
					uint32_t m0_pre = t * mtile;
					uint32_t m1_pre = std::min<uint32_t>(m0_pre + mtile, maps);
					std::vector<int64_t> sums(K, 0);
					for (uint32_t m = m0_pre; m < m1_pre; m++) {
						uint32_t kk2 = 0;
						for (uint32_t c0 = 0; c0 < gi; c0++) {
							for (int32_t kk0 = 0; kk0 < k0; kk0++) {
								for (int32_t kk1 = 0; kk1 < k1; kk1++) {
									uint64_t widx = (((uint64_t)m * gi + c0) * k0 + kk0) * k1 + kk1;
									sums[kk2++] += static_cast<int64_t>(get_tensor_elem_i32(get_W(), widx) - w_zp_const);
								}
							}
						}
					}
					INDT_3 << "{";
					for (uint32_t kk2 = 0; kk2 < K; kk2++) {
						if (kk2 > 0) dst << ", ";
						dst << static_cast<int32_t>(sums[kk2]);
					}
					dst << "}";
					if (t + 1 < tiles_per_group) dst << ",";
					dst << std::endl;
				}
				INDT_2 << "};" << std::endl;
			}
			else {
				INDT_2 << "/* Precompute B tile checksums once per layer/group tile */" << std::endl;
				INDT_2 << "int32_t b_rs_cache[" << tiles_per_group << "][" << K << "];" << std::endl;
				INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
				INDT_3 << "uint32_t m0_pre = m_base + t*MTILE;" << std::endl;
				INDT_3 << "if( m0_pre >= m_limit ) continue;" << std::endl;
				INDT_3 << "uint32_t m1_pre = MIN(m0_pre + MTILE, m_limit);" << std::endl;
				INDT_3 << "for( uint32_t kk2=0; kk2<K; kk2++ ) b_rs_cache[t][kk2] = 0;" << std::endl;
				INDT_3 << "for( uint32_t m=m0_pre; m<m1_pre; m++ ) {" << std::endl;
				INDT_4 << "uint32_t kk2 = 0;" << std::endl;
				INDT_4 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
				INDT_4 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
				INDT_4 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
				INDT_4 << "b_rs_cache[t][kk2++] += ((int32_t)w[m][c0][kk0][kk1] - w_zp);" << std::endl;
				INDT_4 << "}" << std::endl;
				INDT_4 << "}" << std::endl;
				INDT_4 << "}" << std::endl;
				INDT_3 << "}" << std::endl;
				INDT_2 << "}" << std::endl;
			}
			INDT_2 << "int64_t abft_sumC_acc[" << tiles_per_group << "];" << std::endl;
			INDT_2 << "int64_t abft_pred_acc[" << tiles_per_group << "];" << std::endl;
			INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) { abft_sumC_acc[t] = 0; abft_pred_acc[t] = 0; }" << std::endl;
		}

		INDT_2 << "for( int32_t o0=0, i00=" << -pad0 << "; o0<" << out0 << "; o0++, i00+=" << s0 << " ) {" << std::endl;
		INDT_3 << "for( int32_t o1=0, i10=" << -pad1 << "; o1<" << out1 << "; o1++, i10+=" << s1 << " ) {" << std::endl;

		INDT_4 << "/* im2col: A_k = (x - x_zero_point) */" << std::endl;
		INDT_4 << "uint32_t kk = 0;" << std::endl;
		INDT_4 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
		INDT_5 << "uint32_t c = c0" << (group > 1 ? " + gi*g" : "") << ";" << std::endl;
		INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
		INDT_5 << "int32_t i0 = i00 + kk0*" << d0 << ";" << std::endl;
		INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
		INDT_5 << "int32_t i1 = i10 + kk1*" << d1 << ";" << std::endl;
		INDT_5 << "if( i0 >= 0 && i0 < " << in0 << " && i1 >= 0 && i1 < " << in1 << " ) {" << std::endl;
		INDT_5 << "col[kk++] = ((int32_t)x[b][c][i0][i1] - x_zp);" << std::endl;
		INDT_5 << "} else {" << std::endl;
		INDT_5 << "col[kk++] = 0;" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_4 << "}" << std::endl;

		if (checksum_enabled && !options.freivalds_gemm) {
			INDT_4 << "/* ABFT checksum reductions for this GEMM (before tile loop) */" << std::endl;
			INDT_4 << "int64_t lhs_sum = 0;" << std::endl;
			INDT_4 << "for( uint32_t kk2=0; kk2<K; kk2++ ) lhs_sum += (int64_t)col[kk2];" << std::endl;
			INDT_4 << "int64_t pred_cache[" << tiles_per_group << "];" << std::endl;
			INDT_4 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_5 << "pred_cache[t] = 0;" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred_cache[t] += (int64_t)(col[kk2] * b_rs_cache[t][kk2]);" << std::endl;
			INDT_4 << "}" << std::endl;
		}

		INDT_4 << "/* gemm (dot-product): acc32 = sum_k (A_k * B_mk) */" << std::endl;
		INDT_4 << "for( uint32_t m0=" << m_begin << "; m0<" << m_end << "; m0+=MTILE ) {" << std::endl;
		INDT_5 << "uint32_t m1 = MIN(m0 + MTILE, (uint32_t)(" << m_end << "));" << std::endl;

		if (options.abyzft_gemm) {
			INDT_5 << "/* AByzFT: randomized scaling (per A-row, per B-column) */" << std::endl;
			INDT_5 << "uint32_t abyzft_state = (uint32_t)(LAYER_ID ^ (uint32_t)b ^ (uint32_t)o0 ^ (uint32_t)o1 ^ (uint32_t)m0);" << std::endl;
			INDT_5 << "float abyzft_scaleA = 0.25f + 3.75f * ABYZFT_rand01(&abyzft_state);" << std::endl;
			INDT_5 << "float abyzft_scaleB[" << mtile << "];" << std::endl;
			INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) abyzft_scaleB[mi] = 0.25f + 3.75f * ABYZFT_rand01(&abyzft_state);" << std::endl;
			INDT_5 << "double col_scaled[" << K << "];" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) col_scaled[kk2] = (double)col[kk2] * (double)abyzft_scaleA;" << std::endl;
		}

		if (checksum_enabled) {
			if (options.freivalds_gemm) {
				INDT_5 << "uint32_t tile_idx = (m0 - m_base) / MTILE;" << std::endl;
				if (freivalds_checks == 1) {
					INDT_5 << "int64_t freivalds_sumC = 0;" << std::endl;
				}
				else {
					INDT_5 << "int64_t freivalds_sumC[" << freivalds_checks << "];" << std::endl;
					INDT_5 << "for( uint32_t chk=0; chk<" << freivalds_checks << "; chk++ ) freivalds_sumC[chk] = 0;" << std::endl;
				}
			}
			else {
				INDT_5 << "uint32_t tile_idx = (m0 - m_base) / MTILE;" << std::endl;
				INDT_5 << "int32_t* b_rs = b_rs_cache[tile_idx];" << std::endl;
				INDT_5 << "int64_t pred = pred_cache[tile_idx];" << std::endl;
				INDT_5 << "int64_t sumC = 0;" << std::endl;
			}
		}

		INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) {" << std::endl;
		if (options.abyzft_gemm) {
			INDT_5 << "float abyzft_scaleB_m = abyzft_scaleB[m - m0];" << std::endl;
			INDT_5 << "double abyzft_scaleAB = (double)abyzft_scaleA * (double)abyzft_scaleB_m;" << std::endl;
			INDT_5 << "double acc_scaled = 0.0;" << std::endl;
			INDT_5 << "uint32_t k = 0;" << std::endl;
			INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
			INDT_5 << "double rhs_scaled = (double)(((int32_t)w[m][c0][kk0][kk1]) - w_zp) * (double)abyzft_scaleB_m;" << std::endl;
			INDT_5 << "acc_scaled += col_scaled[k++] * rhs_scaled;" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "int32_t acc32 = (abyzft_scaleAB != 0.0) ? (int32_t) llround(acc_scaled / abyzft_scaleAB) : 0;" << std::endl;
		}
		else {
			INDT_5 << "int32_t acc32 = 0;" << std::endl;
			INDT_5 << "uint32_t k = 0;" << std::endl;
			INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
			INDT_5 << "acc32 += col[k++] * (((int32_t)w[m][c0][kk0][kk1]) - w_zp);" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
		}

		INDT_5 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		INDT_5 << "const uint32_t P = " << out0 << "u * " << out1 << "u;" << std::endl;
		INDT_5 << "uint32_t out_idx = ((b * " << maps << "u + m) * P) + ((uint32_t)o0 * " << out1 << "u + (uint32_t)o1);" << std::endl;
		INDT_5 << "int32_t fault_delta = (int32_t) llround((double)FAULT_VALUE);" << std::endl;
		INDT_5 << "if( FAULT_MODEL==0 ) {" << std::endl;
		INDT_5 << "if( out_idx == FAULT_INDEX ) { acc32 += fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "} else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_5 << "uint32_t base_m = (FAULT_INDEX / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = FAULT_INDEX % P;" << std::endl;
		INDT_5 << "int32_t delta = 0;" << std::endl;
		if (group > 1 && go == 1) {
			INDT_5 << "uint32_t base_row = base_p / " << out1 << "u;" << std::endl;
			INDT_5 << "uint32_t base_col = base_p % " << out1 << "u;" << std::endl;
			INDT_5 << "bool ok = (base_m >= m0) && (base_m < m1) && (" << out0 << "u > 1u) && (" << out1 << "u > 1u);" << std::endl;
			INDT_5 << "if( ok ) {" << std::endl;
			INDT_5 << "uint32_t row0 = MIN(base_row, " << out0 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t col0 = MIN(base_col, " << out1 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t b_off = (FAULT_INDEX / (" << maps << "u * P));" << std::endl;
			INDT_5 << "uint32_t base = ((b_off * " << maps << "u + base_m) * P) + (row0 * " << out1 << "u + col0);" << std::endl;
			INDT_5 << "if( out_idx == base ) delta = +fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base + 1u ) delta = -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base + " << out1 << "u ) delta = -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base + " << out1 << "u + 1u ) delta = +fault_delta;" << std::endl;
			INDT_5 << "}" << std::endl;
		} else {
			INDT_5 << "bool ok = true;" << std::endl;
			INDT_5 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
			INDT_5 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
			INDT_5 << "if( ok ) {" << std::endl;
			INDT_5 << "if( out_idx == FAULT_INDEX ) delta = +fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == FAULT_INDEX + P ) delta = -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +fault_delta;" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		INDT_5 << "if( delta != 0 ) { acc32 += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "} else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_5 << "int32_t delta = 0;" << std::endl;
		INDT_5 << "for( uint32_t r=0; r<FAULT_N; r++ ) {" << std::endl;
		INDT_5 << "uint32_t base = FAULT_INDEX + r * FAULT_STRIDE;" << std::endl;
		INDT_5 << "uint32_t base_m = (base / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = base % P;" << std::endl;
		if (group > 1 && go == 1) {
			INDT_5 << "uint32_t base_row = base_p / " << out1 << "u;" << std::endl;
			INDT_5 << "uint32_t base_col = base_p % " << out1 << "u;" << std::endl;
			INDT_5 << "bool ok = (base_m >= m0) && (base_m < m1) && (" << out0 << "u > 1u) && (" << out1 << "u > 1u);" << std::endl;
			INDT_5 << "if( !ok ) continue;" << std::endl;
			INDT_5 << "uint32_t row0 = MIN(base_row, " << out0 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t col0 = MIN(base_col, " << out1 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t b_off = (base / (" << maps << "u * P));" << std::endl;
			INDT_5 << "uint32_t base2 = ((b_off * " << maps << "u + base_m) * P) + (row0 * " << out1 << "u + col0);" << std::endl;
			INDT_5 << "if( out_idx == base2 ) delta += +fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base2 + 1u ) delta += -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base2 + " << out1 << "u ) delta += -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base2 + " << out1 << "u + 1u ) delta += +fault_delta;" << std::endl;
		} else {
			INDT_5 << "bool ok = true;" << std::endl;
			INDT_5 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
			INDT_5 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
			INDT_5 << "if( !ok ) continue;" << std::endl;
			INDT_5 << "if( out_idx == base ) delta += +fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base + 1u ) delta += -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base + P ) delta += -fault_delta;" << std::endl;
			INDT_5 << "else if( out_idx == base + P + 1u ) delta += +fault_delta;" << std::endl;
		}
		INDT_5 << "}" << std::endl;
		INDT_5 << "if( delta != 0 ) { acc32 += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;

		if (checksum_enabled) {
			if (options.freivalds_gemm) {
				if (freivalds_checks == 1) {
					INDT_5 << "freivalds_sumC += (int64_t)acc32 * (int64_t)freivalds_r_cache[0][tile_idx][m-m0];" << std::endl;
				}
				else {
					INDT_5 << "for( uint32_t chk=0; chk<" << freivalds_checks << "; chk++ ) {" << std::endl;
					INDT_5 << "freivalds_sumC[chk] += (int64_t)acc32 * (int64_t)freivalds_r_cache[chk][tile_idx][m-m0];" << std::endl;
					INDT_5 << "}" << std::endl;
				}
			}
			else {
				INDT_5 << "sumC += (int64_t)acc32;" << std::endl;
			}
		}

		INDT_5 << "y[b][m][o0][o1] = acc32;" << std::endl;
		INDT_5 << "} /* m */" << std::endl;

		if (checksum_enabled) {
			if (options.freivalds_gemm) {
				INDT_5 << "/* Freivalds verify (integer domain): r^T C_tile == A^T (B_tile r) */" << std::endl;
				if (freivalds_checks == 1) {
					INDT_5 << "int64_t pred = 0;" << std::endl;
					INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += (int64_t)(col[kk2] * freivalds_brs_cache[0][tile_idx][kk2]);" << std::endl;
					INDT_5 << "if( freivalds_sumC != pred ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
				}
				else {
					INDT_5 << "for( uint32_t chk=0; chk<" << freivalds_checks << "; chk++ ) {" << std::endl;
					INDT_5 << "int64_t pred = 0;" << std::endl;
					INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += (int64_t)(col[kk2] * freivalds_brs_cache[chk][tile_idx][kk2]);" << std::endl;
					INDT_5 << "if( freivalds_sumC[chk] != pred ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
					INDT_5 << "}" << std::endl;
				}
			}
			else {
				INDT_5 << "/* ABFT verify (integer domain): sum(C_tile) == A * (B_tile 1) */" << std::endl;
				INDT_5 << "abft_sumC_acc[tile_idx] += sumC;" << std::endl;
				INDT_5 << "abft_pred_acc[tile_idx] += pred;" << std::endl;
			}
		}

		INDT_4 << "} /* m0 */" << std::endl;
		INDT_3 << "} /* o1 */" << std::endl;
		INDT_2 << "} /* o0 */" << std::endl;
		if (checksum_enabled && !options.freivalds_gemm) {
			INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_3 << "if( abft_sumC_acc[t] != abft_pred_acc[t] ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
			INDT_2 << "}" << std::endl;
		}
		if (group > 1)
			INDT_1 << "} /* g */" << std::endl;
		INDT_1 << "} /* b */" << std::endl;
	}
};
} // namespace toC
