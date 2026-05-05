/* This file is part of onnx2c.
 *
 * QLinearConv
 * Quantized Convolution
 */

#include "spatialfilter.h"
#include "../options.h"

namespace toC {
class QLinearConv : public SpatialFilter {
	public:
	QLinearConv()
	{
		op_name = "QLinearConv";
	}

	const Tensor* get_X(void) const override { return get_input_tensor(0); }
	const Tensor* get_W(void) const override { return get_input_tensor(3); }

	void name_scalar_input(unsigned input_no, std::string name)
	{
		name_input(input_no, name);
		if (!(get_input_tensor(input_no)->data_dim.size() == 0 ||
		      (get_input_tensor(input_no)->data_dim.size() == 1 && get_input_tensor(input_no)->data_dim[0] == 1))) {
			ERROR(name << " must be scalar");
		}
	}

	void print_output_cell_init(std::ostream& dst, const std::string&) const override
	{
		INDT_3 << "int32_t a = ";
		if (get_number_of_inputs() < 9)
			dst << "0";
		else
			dst << "bias[m]";
		dst << ";" << std::endl;
	}

	void print_output_cell_calc(std::ostream& dst,
	                            const std::string& x_idx,
	                            const std::string& w_idx,
	                            const std::string&) const override
	{
		INDT_4 << "a += ((int32_t)x" << x_idx << " - x_zero_point[0]) * ((int32_t)w" << w_idx << " - w_zero_point[0]);" << std::endl;
	}

	void print_output_cell_finalize(std::ostream& dst, const std::string& y_idx) const override
	{
		std::string float_dtype = get_input_tensor(1)->data_type_str();
		auto [lower, upper] = get_output_tensor(0)->get_type_bounds();
		INDT_3 << float_dtype << " scaled = ((" << float_dtype << ")a) * (x_scale[0] * w_scale[0]) / y_scale[0];" << std::endl;
		INDT_3 << "scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
		INDT_3 << "int32_t q = (int32_t) llround((double)scaled);" << std::endl;
		INDT_3 << "q = MIN(MAX(q, " << lower << "), " << upper << ");" << std::endl;
		INDT_3 << "y" << y_idx << " = (" << get_output_tensor(0)->data_type_str() << ") q;" << std::endl;
	}

	void print_im2col_matmul(std::ostream& dst) const
	{
		const auto* x = get_X();
		const auto* y = get_output_tensor(0);
		const uint32_t n_data_dims = get_numDataDim();

		if (n_data_dims != 2) {
			LOG(WARNING) << "QLinearConv im2col backend only supports 2D conv; falling back to direct loops for " << onnx_name << std::endl;
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

		INDT_1 << "/* QLinearConv (im2col + dot-product) */" << std::endl;
		INDT_1 << "const uint32_t LAYER_ID = " << sweep_layer_id << ";" << std::endl;
		INDT_1 << "const uint32_t K = " << K << ";" << std::endl;
		INDT_1 << "int32_t col[" << K << "];" << std::endl;

		INDT_1 << "int32_t x_zp = (int32_t)x_zero_point[0];" << std::endl;
		INDT_1 << "int32_t w_zp = (int32_t)w_zero_point[0];" << std::endl;

		INDT_1 << "for( uint32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		if (group > 1) {
			INDT_1 << "uint32_t go = " << go << "; // output group size, i.e. maps/group" << std::endl;
			INDT_1 << "uint32_t gi = " << gi << "; // input group size, i.e. channels/group" << std::endl;
			INDT_1 << "for( uint32_t g=0; g<" << group << "; g++ ) {" << std::endl;
		}

		const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 16;
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || options.freivalds_gemm || options.gvfa_gemm;
		const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
		const bool freivalds_enabled = options.freivalds_gemm;
		const bool gvfa_enabled = options.gvfa_gemm;
		const uint32_t randomized_checks = freivalds_enabled
		    ? (options.freivalds_checks ? options.freivalds_checks : 1)
		    : (options.gvfa_checks ? options.gvfa_checks : 1);

		std::string m_begin = (group > 1) ? "go*g" : "0";
		std::string m_end = (group > 1) ? "go*(g+1)" : std::to_string(maps);
		const uint32_t tiles_per_group = ((group > 1 ? go : maps) + mtile - 1) / mtile;
		std::string float_dtype = get_input_tensor(1)->data_type_str();
		const bool unsigned_quant = typeConstraint_unsigned_integers(get_X());
		const char* scale_picker = unsigned_quant ? "ABYZFT_pick_scale_u8" : "ABYZFT_pick_scale_s8";
		auto [lower, upper] = get_output_tensor(0)->get_type_bounds();
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
		bool use_compiletime_abyzft_wbase = false;
		int32_t w_zp_const = 0;
		if (options.abyzft_gemm && get_W()->isConst) {
			use_compiletime_abyzft_wbase = supports_compiletime_i32_read(get_W());
			const Tensor* w_zp_t = get_input_tensor(5);
			if (!w_zp_t->isConst || !supports_compiletime_i32_read(w_zp_t))
				use_compiletime_abyzft_wbase = false;
			else
				w_zp_const = get_tensor_elem_i32(w_zp_t, 0);
		}
		bool use_abyzft_i32_accum = false;
		if (options.abyzft_gemm) {
			const Tensor* x_zp_t = get_input_tensor(2);
			const Tensor* w_zp_t = get_input_tensor(5);
			auto abs64 = [](int64_t v) -> int64_t { return v < 0 ? -v : v; };
			auto type_bounds_i64 = [](const Tensor* t) -> std::pair<int64_t, int64_t> {
				switch (t->data_type) {
					case onnx::TensorProto_DataType_INT8: return {INT8_MIN, INT8_MAX};
					case onnx::TensorProto_DataType_UINT8: return {0, UINT8_MAX};
					case onnx::TensorProto_DataType_INT16: return {INT16_MIN, INT16_MAX};
					case onnx::TensorProto_DataType_UINT16: return {0, UINT16_MAX};
					case onnx::TensorProto_DataType_INT32: return {INT32_MIN, INT32_MAX};
					case onnx::TensorProto_DataType_UINT32: return {0, INT64_C(4294967295)};
					default: return {INT64_MIN / 4, INT64_MAX / 4};
				}
			};
			if (x_zp_t->isConst && w_zp_t->isConst &&
			    supports_compiletime_i32_read(x_zp_t) &&
			    supports_compiletime_i32_read(w_zp_t)) {
				int32_t x_zp_const = get_tensor_elem_i32(x_zp_t, 0);
				int32_t w_zp_bound = get_tensor_elem_i32(w_zp_t, 0);
				auto [x_lo, x_hi] = type_bounds_i64(get_X());
				auto [w_lo, w_hi] = type_bounds_i64(get_W());
				const int64_t max_abs_x = std::max(abs64((int64_t)x_lo - x_zp_const), abs64((int64_t)x_hi - x_zp_const));
				const int64_t max_abs_w = std::max(abs64((int64_t)w_lo - w_zp_bound), abs64((int64_t)w_hi - w_zp_bound));
				const int64_t max_scale = 8;
				const int64_t worst = (int64_t)K * (max_abs_x * max_scale) * (max_abs_w * max_scale);
				use_abyzft_i32_accum = (worst <= 2147483647ll);
			}
		}

		INDT_2 << "const uint32_t MTILE = " << mtile << ";" << std::endl;
		INDT_2 << "const uint32_t m_base = (uint32_t)(" << m_begin << ");" << std::endl;
		INDT_2 << "const uint32_t m_limit = (uint32_t)(" << m_end << ");" << std::endl;
		if (checksum_enabled && randomized_enabled) {
			INDT_2 << "float randomized_r_cache[" << randomized_checks << "][" << tiles_per_group << "][" << mtile << "];" << std::endl;
			INDT_2 << "double randomized_brs_cache[" << randomized_checks << "][" << tiles_per_group << "][" << K << "];" << std::endl;
			INDT_2 << "for( uint32_t chk=0; chk<" << randomized_checks << "; chk++ ) {" << std::endl;
			INDT_3 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_4 << "uint32_t m0_pre = m_base + t*MTILE;" << std::endl;
			INDT_4 << "if( m0_pre >= m_limit ) continue;" << std::endl;
			INDT_4 << "uint32_t m1_pre = MIN(m0_pre + MTILE, m_limit);" << std::endl;
			if (freivalds_enabled) {
				INDT_4 << "uint32_t rand_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ (uint32_t)b ^ (uint32_t)m0_pre ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
				INDT_4 << "uint32_t r_any = 0;" << std::endl;
				INDT_4 << "for( uint32_t mi=0; mi<(m1_pre-m0_pre); mi++ ) { uint32_t bit = ABYZFT_randbit(&rand_state); randomized_r_cache[chk][t][mi] = (float)bit; r_any |= bit; }" << std::endl;
				INDT_4 << "if( !r_any && (m1_pre>m0_pre) ) randomized_r_cache[chk][t][0] = 1.0f;" << std::endl;
			}
			if (gvfa_enabled) {
				INDT_4 << "uint32_t rand_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ (uint32_t)b ^ (uint32_t)m0_pre ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
				INDT_4 << "for( uint32_t mi=0; mi<(m1_pre-m0_pre); mi++ ) randomized_r_cache[chk][t][mi] = ABYZFT_rand01(&rand_state);" << std::endl;
			}
			INDT_4 << "for( uint32_t kk2=0; kk2<K; kk2++ ) randomized_brs_cache[chk][t][kk2] = 0.0;" << std::endl;
			INDT_4 << "for( uint32_t m=m0_pre; m<m1_pre; m++ ) {" << std::endl;
			if (freivalds_enabled)
				INDT_5 << "if( randomized_r_cache[chk][t][m-m0_pre] == 0.0f ) continue;" << std::endl;
			else
				INDT_5 << "float r = randomized_r_cache[chk][t][m-m0_pre];" << std::endl;
			INDT_5 << "uint32_t kk2 = 0;" << std::endl;
			INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
			if (freivalds_enabled)
				INDT_5 << "randomized_brs_cache[chk][t][kk2++] += (double)(((int32_t)w[m][c0][kk0][kk1]) - w_zp);" << std::endl;
			else
				INDT_5 << "randomized_brs_cache[chk][t][kk2++] += (double)r * (double)(((int32_t)w[m][c0][kk0][kk1]) - w_zp);" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_4 << "}" << std::endl;
			INDT_3 << "}" << std::endl;
			INDT_2 << "}" << std::endl;
		}
		if (checksum_enabled && !randomized_enabled) {
			INDT_2 << "int32_t b_rs_cache[" << tiles_per_group << "][" << K << "];" << std::endl;
			INDT_2 << "int64_t bias_sum_cache[" << tiles_per_group << "];" << std::endl;
			INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_3 << "uint32_t m0_pre = m_base + t*MTILE;" << std::endl;
			INDT_3 << "if( m0_pre >= m_limit ) continue;" << std::endl;
			INDT_3 << "uint32_t m1_pre = MIN(m0_pre + MTILE, m_limit);" << std::endl;
			INDT_3 << "for( uint32_t kk2=0; kk2<K; kk2++ ) b_rs_cache[t][kk2] = 0;" << std::endl;
			INDT_3 << "bias_sum_cache[t] = 0;" << std::endl;
			INDT_3 << "for( uint32_t m=m0_pre; m<m1_pre; m++ ) {" << std::endl;
			if (get_number_of_inputs() == 9)
				INDT_4 << "bias_sum_cache[t] += (int64_t)bias[m];" << std::endl;
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
			INDT_2 << "int64_t abft_sumC_acc[" << tiles_per_group << "];" << std::endl;
			INDT_2 << "int64_t abft_pred_acc[" << tiles_per_group << "];" << std::endl;
			INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) { abft_sumC_acc[t] = 0; abft_pred_acc[t] = 0; }" << std::endl;
		}

		INDT_2 << "for( int32_t o0=0, i00=" << -pad0 << "; o0<" << out0 << "; o0++, i00+=" << s0 << " ) {" << std::endl;
		INDT_3 << "for( int32_t o1=0, i10=" << -pad1 << "; o1<" << out1 << "; o1++, i10+=" << s1 << " ) {" << std::endl;
		INDT_4 << "uint32_t kk = 0;" << std::endl;
		INDT_4 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
		INDT_5 << "uint32_t c = c0" << (group > 1 ? " + gi*g" : "") << ";" << std::endl;
		INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
		INDT_5 << "int32_t i0 = i00 + kk0*" << d0 << ";" << std::endl;
		INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
		INDT_5 << "int32_t i1 = i10 + kk1*" << d1 << ";" << std::endl;
		INDT_5 << "if( i0 >= 0 && i0 < " << in0 << " && i1 >= 0 && i1 < " << in1 << " ) {" << std::endl;
		INDT_6 << "col[kk++] = ((int32_t)x[b][c][i0][i1] - x_zp);" << std::endl;
		INDT_5 << "} else {" << std::endl;
		INDT_6 << "col[kk++] = 0;" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_4 << "}" << std::endl;

		if (checksum_enabled && !randomized_enabled) {
			INDT_4 << "int64_t pred_cache[" << tiles_per_group << "];" << std::endl;
			INDT_4 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_5 << "pred_cache[t] = bias_sum_cache[t];" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred_cache[t] += (int64_t)(col[kk2] * b_rs_cache[t][kk2]);" << std::endl;
			INDT_4 << "}" << std::endl;
		}

		INDT_4 << "for( uint32_t m0=" << m_begin << "; m0<" << m_end << "; m0+=MTILE ) {" << std::endl;
		INDT_5 << "uint32_t m1 = MIN(m0 + MTILE, (uint32_t)(" << m_end << "));" << std::endl;

		if (options.abyzft_gemm) {
			INDT_5 << "uint32_t abyzft_state = (uint32_t)(LAYER_ID ^ (uint32_t)b ^ (uint32_t)o0 ^ (uint32_t)o1 ^ (uint32_t)m0);" << std::endl;
			INDT_5 << "int32_t abyzft_scaleA = (int32_t)" << scale_picker << "(&abyzft_state);" << std::endl;
			INDT_5 << "int32_t abyzft_scaleB[" << mtile << "];" << std::endl;
			INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) abyzft_scaleB[mi] = (int32_t)" << scale_picker << "(&abyzft_state);" << std::endl;
			INDT_5 << "int16_t col_scaled[" << K << "];" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) col_scaled[kk2] = (int16_t)(col[kk2] * abyzft_scaleA);" << std::endl;
			if (use_compiletime_abyzft_wbase) {
				INDT_5 << "static const int16_t w_base_cache[" << maps << "][" << K << "] = {" << std::endl;
				for (uint32_t m = 0; m < maps; m++) {
					INDT_6 << "{";
					uint32_t kk2 = 0;
					for (uint32_t c0 = 0; c0 < gi; c0++) {
						for (int32_t kk0 = 0; kk0 < k0; kk0++) {
							for (int32_t kk1 = 0; kk1 < k1; kk1++) {
								uint64_t widx = (((uint64_t)m * gi + c0) * k0 + kk0) * k1 + kk1;
								if (kk2++ > 0) dst << ", ";
								dst << static_cast<int16_t>(get_tensor_elem_i32(get_W(), widx) - w_zp_const);
							}
						}
					}
					dst << "}";
					if (m + 1 < maps) dst << ",";
					dst << std::endl;
				}
				INDT_5 << "};" << std::endl;
			}
			INDT_5 << "int16_t w_scaled_tile[" << mtile << "][" << K << "];" << std::endl;
			INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) {" << std::endl;
			INDT_6 << "uint32_t m_scaled = m0 + mi;" << std::endl;
			INDT_6 << "int32_t scaleB = abyzft_scaleB[mi];" << std::endl;
			if (use_compiletime_abyzft_wbase) {
				INDT_6 << "const int16_t* w_base = w_base_cache[m_scaled];" << std::endl;
				INDT_6 << "for( uint32_t kk2=0; kk2<K; kk2++ ) w_scaled_tile[mi][kk2] = (int16_t)(w_base[kk2] * scaleB);" << std::endl;
			}
			else {
				INDT_6 << "uint32_t kk2 = 0;" << std::endl;
				INDT_6 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
				INDT_6 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
				INDT_6 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
				INDT_6 << "int32_t wq = (((int32_t)w[m_scaled][c0][kk0][kk1]) - w_zp);" << std::endl;
				INDT_6 << "w_scaled_tile[mi][kk2++] = (int16_t)(wq * scaleB);" << std::endl;
				INDT_6 << "}" << std::endl;
				INDT_6 << "}" << std::endl;
				INDT_6 << "}" << std::endl;
			}
			INDT_5 << "}" << std::endl;
		}

		if (checksum_enabled) {
			if (!randomized_enabled) {
				INDT_5 << "uint32_t tile_idx = (m0 - m_base) / MTILE;" << std::endl;
				INDT_5 << "int64_t pred = pred_cache[tile_idx];" << std::endl;
			}
			else
				INDT_5 << "uint32_t tile_idx = (m0 - m_base) / MTILE;" << std::endl;
			INDT_5 << "int32_t acc_tile[" << mtile << "];" << std::endl;
		}

		INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) {" << std::endl;
		if (options.abyzft_gemm) {
			INDT_6 << "int32_t abyzft_scaleB_m = abyzft_scaleB[m - m0];" << std::endl;
			INDT_6 << "int64_t abyzft_scaleAB = (int64_t)abyzft_scaleA * (int64_t)abyzft_scaleB_m;" << std::endl;
			INDT_6 << "int64_t abyzft_scaleAB_mag = (abyzft_scaleAB < 0) ? -abyzft_scaleAB : abyzft_scaleAB;" << std::endl;
			INDT_6 << "int32_t abyzft_scaleAB_sign = (abyzft_scaleAB < 0) ? -1 : 1;" << std::endl;
			INDT_6 << "uint32_t abyzft_shiftAB = ABYZFT_pow2_shift_u64((uint64_t)abyzft_scaleAB_mag);" << std::endl;
			if (use_abyzft_i32_accum)
				INDT_6 << "int32_t acc_scaled = 0;" << std::endl;
			else
				INDT_6 << "int64_t acc_scaled = 0;" << std::endl;
			if (get_number_of_inputs() < 9)
				INDT_6 << "int32_t acc32 = 0;" << std::endl;
			else
				INDT_6 << "int32_t acc32 = bias[m];" << std::endl;
			INDT_6 << "const int16_t* w_scaled = w_scaled_tile[m - m0];" << std::endl;
			if (use_abyzft_i32_accum)
				INDT_6 << "for( uint32_t kk2=0; kk2<K; kk2++ ) acc_scaled += (int32_t)col_scaled[kk2] * (int32_t)w_scaled[kk2];" << std::endl;
			else
				INDT_6 << "for( uint32_t kk2=0; kk2<K; kk2++ ) acc_scaled += (int64_t)col_scaled[kk2] * (int64_t)w_scaled[kk2];" << std::endl;
			if (use_abyzft_i32_accum)
				INDT_6 << "int64_t acc_scaled_fault = (int64_t)acc_scaled;" << std::endl;
		}
		else {
			INDT_6 << "int32_t acc32 = ";
			if (get_number_of_inputs() < 9)
				dst << "0;" << std::endl;
			else
				dst << "bias[m];" << std::endl;
			INDT_6 << "int32_t acc32_check = acc32;" << std::endl;
			INDT_6 << "uint32_t k = 0;" << std::endl;
			INDT_6 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
			INDT_6 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
			INDT_6 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
			INDT_6 << "acc32 += col[k++] * (((int32_t)w[m][c0][kk0][kk1]) - w_zp);" << std::endl;
			INDT_6 << "}" << std::endl;
			INDT_6 << "}" << std::endl;
			INDT_6 << "}" << std::endl;
			INDT_6 << "acc32_check = acc32;" << std::endl;
		}

		INDT_6 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		INDT_6 << "const uint32_t P = " << out0 << "u * " << out1 << "u;" << std::endl;
		INDT_6 << "uint32_t out_idx = ((b * " << maps << "u + m) * P) + ((uint32_t)o0 * " << out1 << "u + (uint32_t)o1);" << std::endl;
		INDT_6 << "double eff_scale = (double)x_scale[0] * (double)w_scale[0];" << std::endl;
		INDT_6 << "int32_t fault_delta = 0;" << std::endl;
		INDT_6 << "if( eff_scale != 0.0 ) fault_delta = (int32_t) llround((double)FAULT_VALUE / eff_scale);" << std::endl;
		if (options.abyzft_gemm)
			INDT_6 << "int64_t scaled_fault_delta = (int64_t)fault_delta;" << std::endl;
		INDT_6 << "if( FAULT_MODEL==0 ) {" << std::endl;
		if (options.abyzft_gemm)
			INDT_6 << "if( out_idx == FAULT_INDEX ) { " << (use_abyzft_i32_accum ? "acc_scaled_fault" : "acc_scaled") << " += scaled_fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_6 << "if( out_idx == FAULT_INDEX ) { acc32 += fault_delta; acc32_check += fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_6 << "} else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_6 << "uint32_t base_m = (FAULT_INDEX / P) % " << maps << "u;" << std::endl;
		INDT_6 << "uint32_t base_p = FAULT_INDEX % P;" << std::endl;
		INDT_6 << "int32_t delta = 0;" << std::endl;
		if (group > 1 && go == 1) {
			INDT_6 << "uint32_t base_row = base_p / " << out1 << "u;" << std::endl;
			INDT_6 << "uint32_t base_col = base_p % " << out1 << "u;" << std::endl;
			INDT_6 << "bool ok = (base_m >= m0) && (base_m < m1) && (" << out0 << "u > 1u) && (" << out1 << "u > 1u);" << std::endl;
			INDT_6 << "if( ok ) {" << std::endl;
			INDT_6 << "uint32_t row0 = MIN(base_row, " << out0 << "u - 2u);" << std::endl;
			INDT_6 << "uint32_t col0 = MIN(base_col, " << out1 << "u - 2u);" << std::endl;
			INDT_6 << "uint32_t b_off = (FAULT_INDEX / (" << maps << "u * P));" << std::endl;
			INDT_6 << "uint32_t base = ((b_off * " << maps << "u + base_m) * P) + (row0 * " << out1 << "u + col0);" << std::endl;
			INDT_6 << "if( out_idx == base ) delta = +fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base + 1u ) delta = -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base + " << out1 << "u ) delta = -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base + " << out1 << "u + 1u ) delta = +fault_delta;" << std::endl;
			INDT_6 << "}" << std::endl;
		}
		else {
			INDT_6 << "bool ok = true;" << std::endl;
			INDT_6 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
			INDT_6 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
			INDT_6 << "if( ok ) {" << std::endl;
			INDT_6 << "if( out_idx == FAULT_INDEX ) delta = +fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == FAULT_INDEX + P ) delta = -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +fault_delta;" << std::endl;
			INDT_6 << "}" << std::endl;
		}
		if (options.abyzft_gemm)
			INDT_6 << "if( delta != 0 ) { " << (use_abyzft_i32_accum ? "acc_scaled_fault" : "acc_scaled") << " += (int64_t)delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_6 << "if( delta != 0 ) { acc32 += delta; acc32_check += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_6 << "} else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_6 << "int32_t delta = 0;" << std::endl;
		INDT_6 << "for( uint32_t r=0; r<FAULT_N; r++ ) {" << std::endl;
		INDT_6 << "uint32_t base = FAULT_INDEX + r * FAULT_STRIDE;" << std::endl;
		INDT_6 << "uint32_t base_m = (base / P) % " << maps << "u;" << std::endl;
		INDT_6 << "uint32_t base_p = base % P;" << std::endl;
		if (group > 1 && go == 1) {
			INDT_6 << "uint32_t base_row = base_p / " << out1 << "u;" << std::endl;
			INDT_6 << "uint32_t base_col = base_p % " << out1 << "u;" << std::endl;
			INDT_6 << "bool ok = (base_m >= m0) && (base_m < m1) && (" << out0 << "u > 1u) && (" << out1 << "u > 1u);" << std::endl;
			INDT_6 << "if( !ok ) continue;" << std::endl;
			INDT_6 << "uint32_t row0 = MIN(base_row, " << out0 << "u - 2u);" << std::endl;
			INDT_6 << "uint32_t col0 = MIN(base_col, " << out1 << "u - 2u);" << std::endl;
			INDT_6 << "uint32_t b_off = (base / (" << maps << "u * P));" << std::endl;
			INDT_6 << "uint32_t base2 = ((b_off * " << maps << "u + base_m) * P) + (row0 * " << out1 << "u + col0);" << std::endl;
			INDT_6 << "if( out_idx == base2 ) delta += +fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base2 + 1u ) delta += -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base2 + " << out1 << "u ) delta += -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base2 + " << out1 << "u + 1u ) delta += +fault_delta;" << std::endl;
		}
		else {
			INDT_6 << "bool ok = true;" << std::endl;
			INDT_6 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
			INDT_6 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
			INDT_6 << "if( !ok ) continue;" << std::endl;
			INDT_6 << "if( out_idx == base ) delta += +fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base + 1u ) delta += -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base + P ) delta += -fault_delta;" << std::endl;
			INDT_6 << "else if( out_idx == base + P + 1u ) delta += +fault_delta;" << std::endl;
		}
		INDT_6 << "}" << std::endl;
		if (options.abyzft_gemm)
			INDT_6 << "if( delta != 0 ) { " << (use_abyzft_i32_accum ? "acc_scaled_fault" : "acc_scaled") << " += (int64_t)delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_6 << "if( delta != 0 ) { acc32 += delta; acc32_check += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_6 << "}" << std::endl;
		INDT_6 << "}" << std::endl;
		if (options.abyzft_gemm) {
			if (get_number_of_inputs() < 9) {
				if (use_abyzft_i32_accum)
					INDT_6 << "if( abyzft_scaleAB_mag != 0 ) acc32 = abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled_fault, abyzft_shiftAB);" << std::endl;
				else
					INDT_6 << "if( abyzft_scaleAB_mag != 0 ) acc32 = (int32_t)(abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled, abyzft_shiftAB));" << std::endl;
			}
			else {
				if (use_abyzft_i32_accum)
					INDT_6 << "if( abyzft_scaleAB_mag != 0 ) acc32 += abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled_fault, abyzft_shiftAB);" << std::endl;
				else
					INDT_6 << "if( abyzft_scaleAB_mag != 0 ) acc32 += (int32_t)(abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled, abyzft_shiftAB));" << std::endl;
			}
		}

		if (checksum_enabled)
			INDT_6 << "acc_tile[m-m0] = " << (options.abyzft_gemm ? "acc32" : "acc32_check") << ";" << std::endl;

		INDT_6 << float_dtype << " scaled = ((" << float_dtype << ")acc32) * (x_scale[0] * w_scale[0]) / y_scale[0];" << std::endl;
		INDT_6 << "scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
		INDT_6 << "int32_t q = (int32_t) llround((double)scaled);" << std::endl;
		INDT_6 << "q = MIN(MAX(q, " << lower << "), " << upper << ");" << std::endl;
		INDT_6 << "y[b][m][o0][o1] = (" << get_output_tensor(0)->data_type_str() << ") q;" << std::endl;
		INDT_5 << "} /* m */" << std::endl;

		if (checksum_enabled) {
			if (randomized_enabled) {
				INDT_5 << "/* Randomized verify: r^T C_tile == A^T (B_tile r) */" << std::endl;
				if (randomized_checks == 1) {
						INDT_5 << "double randomized_sumC = 0.0;" << std::endl;
						INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) {" << std::endl;
						if (freivalds_enabled)
							INDT_6 << "if( randomized_r_cache[0][tile_idx][mi] != 0.0f ) randomized_sumC += (double)acc_tile[mi];" << std::endl;
						else
							INDT_6 << "randomized_sumC += (double)acc_tile[mi] * (double)randomized_r_cache[0][tile_idx][mi];" << std::endl;
						INDT_5 << "}" << std::endl;
						if (get_number_of_inputs() == 9)
							INDT_5 << "double bias_sum = 0.0;" << std::endl;
						else
							INDT_5 << "double bias_sum = 0.0;" << std::endl;
						INDT_5 << "double pred = 0.0;" << std::endl;
						INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += (double)col[kk2] * randomized_brs_cache[0][tile_idx][kk2];" << std::endl;
						if (get_number_of_inputs() == 9) {
							if (freivalds_enabled)
								INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) if( randomized_r_cache[0][tile_idx][m-m0] != 0.0f ) bias_sum += (double)bias[m];" << std::endl;
							else
								INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) bias_sum += (double)bias[m] * (double)randomized_r_cache[0][tile_idx][m-m0];" << std::endl;
						}
						INDT_5 << "pred += bias_sum;" << std::endl;
					if (gvfa_enabled)
						INDT_5 << "if( fabs(randomized_sumC - pred) > (double)" << options.abft_eps << " ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
					else
						INDT_5 << "if( randomized_sumC != pred ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
				}
				else {
					INDT_5 << "for( uint32_t chk=0; chk<" << randomized_checks << "; chk++ ) {" << std::endl;
					INDT_6 << "double randomized_sumC = 0.0;" << std::endl;
					INDT_6 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) {" << std::endl;
					if (freivalds_enabled)
						INDT_6 << "if( randomized_r_cache[chk][tile_idx][mi] != 0.0f ) randomized_sumC += (double)acc_tile[mi];" << std::endl;
					else
						INDT_6 << "randomized_sumC += (double)acc_tile[mi] * (double)randomized_r_cache[chk][tile_idx][mi];" << std::endl;
					INDT_6 << "}" << std::endl;
					INDT_6 << "double bias_sum = 0.0;" << std::endl;
					INDT_6 << "double pred = 0.0;" << std::endl;
					INDT_6 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += (double)col[kk2] * randomized_brs_cache[chk][tile_idx][kk2];" << std::endl;
					if (get_number_of_inputs() == 9) {
						if (freivalds_enabled)
							INDT_6 << "for( uint32_t m=m0; m<m1; m++ ) if( randomized_r_cache[chk][tile_idx][m-m0] != 0.0f ) bias_sum += (double)bias[m];" << std::endl;
						else
							INDT_6 << "for( uint32_t m=m0; m<m1; m++ ) bias_sum += (double)bias[m] * (double)randomized_r_cache[chk][tile_idx][m-m0];" << std::endl;
					}
					INDT_6 << "pred += bias_sum;" << std::endl;
					if (gvfa_enabled)
						INDT_6 << "if( fabs(randomized_sumC - pred) > (double)" << options.abft_eps << " ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
					else
						INDT_6 << "if( randomized_sumC != pred ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
					INDT_5 << "}" << std::endl;
				}
			}
			else {
				INDT_5 << "int64_t sumC = 0;" << std::endl;
				INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) sumC += (int64_t)acc_tile[mi];" << std::endl;
				INDT_5 << "abft_sumC_acc[tile_idx] += sumC;" << std::endl;
				INDT_5 << "abft_pred_acc[tile_idx] += pred;" << std::endl;
			}
		}

		INDT_4 << "} /* m0 */" << std::endl;
		INDT_3 << "} /* o1 */" << std::endl;
		INDT_2 << "} /* o0 */" << std::endl;
		if (checksum_enabled && !randomized_enabled) {
			INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_3 << "if( abft_sumC_acc[t] != abft_pred_acc[t] ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
			INDT_2 << "}" << std::endl;
		}
		if (group > 1)
			INDT_1 << "} /* g */" << std::endl;
		INDT_1 << "} /* b */" << std::endl;
	}

	void print(std::ostream& dst) const override
	{
		print_header_info_comment(dst);
		print_loop_with_padding_checks(dst);
	}

	void resolve() override
	{
		name_input(0, "x");
		name_scalar_input(1, "x_scale");
		name_scalar_input(2, "x_zero_point");

		name_input(3, "w");
		name_scalar_input(4, "w_scale");
		name_scalar_input(5, "w_zero_point");

		name_scalar_input(6, "y_scale");
		name_scalar_input(7, "y_zero_point");

		if (get_number_of_inputs() == 9)
			name_input(8, "bias");

		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		Tensor* rv = new Tensor;
		rv->data_dim = resolve_output_size();
		rv->data_type = get_X()->data_type;
		register_output(rv, "y");
	}
};
} // namespace toC
