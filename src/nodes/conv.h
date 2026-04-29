/* This file is part of onnx2c.
 *
 * Conv
 * Calculates an "industry standard" convolution filter.
 */

#include "spatialfilter.h"
#include "../options.h"
namespace toC {

class Conv : public SpatialFilter {
	public:
	Conv()
	{
		op_name = "Conv";
	}

	void print_im2col_matmul(std::ostream& dst) const
	{
		// Implemented for 1D (rank-3: NCW) and 2D (rank-4: NCHW) conv.
		// Other ranks fall back to the original direct-loop implementation.
		const uint32_t n_data_dims = get_numDataDim();
		if (n_data_dims != 1 && n_data_dims != 2) {
			LOG(WARNING) << "Conv im2col backend only supports 1D/2D conv; falling back to direct loops for " << onnx_name << std::endl;
			print_loop_with_padding_checks(dst);
			return;
		}

		const auto* x = get_X();
		const auto* y = get_Y();

		const uint32_t batch_size = x->data_dim[0];
		const uint32_t channels = x->data_dim[1];
		const uint32_t maps = y->data_dim[1];

		const int32_t in0 = x->data_dim[2];
		const int32_t out0 = y->data_dim[2];
		const int32_t k0 = kernel_shape[0];
		const int32_t s0 = strides[0];
		const int32_t d0 = dilations[0];
		const int32_t pad0 = pads[0];

		int32_t in1 = 1, out1 = 1, k1 = 1, s1 = 1, d1 = 1, pad1 = 0;
		if (n_data_dims == 2) {
			in1 = x->data_dim[3];
			out1 = y->data_dim[3];
			k1 = kernel_shape[1];
			s1 = strides[1];
			d1 = dilations[1];
			pad1 = pads[1];
		}

		const uint32_t go = (group > 0) ? (maps / group) : maps;
		const uint32_t gi = (group > 0) ? (channels / group) : channels;

		// Flattened patch size per output point (per group)
		const uint32_t K = (n_data_dims == 1) ? (gi * k0) : (gi * k0 * k1);

		INDT_1 << "/* Conv (im2col + dot-product) */" << std::endl;
		INDT_1 << "const uint32_t LAYER_ID = " << sweep_layer_id << ";" << std::endl;
		INDT_1 << "const uint32_t K = " << K << ";" << std::endl;
		INDT_1 << x->data_type_str() << " col[" << K << "];" << std::endl;

		INDT_1 << "for( uint32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		if (group > 1) {
			INDT_1 << "uint32_t go = " << go << "; // output group size, i.e. maps/group" << std::endl;
			INDT_1 << "uint32_t gi = " << gi << "; // input group size, i.e. channels/group" << std::endl;
			INDT_1 << "for( uint32_t g=0; g<" << group << "; g++ ) {" << std::endl;
		}

		const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 16;
		const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || randomized_enabled;
		const bool freivalds_enabled = options.freivalds_gemm;
		const bool gvfa_enabled = options.gvfa_gemm;
		const uint32_t randomized_checks = freivalds_enabled
		    ? (options.freivalds_checks ? options.freivalds_checks : 1)
		    : (options.gvfa_checks ? options.gvfa_checks : 1);
		std::string m_begin = (group > 1) ? "go*g" : "0";
		std::string m_end = (group > 1) ? "go*(g+1)" : std::to_string(maps);
		const uint32_t tiles_per_group = ((group > 1 ? go : maps) + mtile - 1) / mtile;

		INDT_2 << "const uint32_t MTILE = " << mtile << ";" << std::endl;
		if (checksum_enabled && !randomized_enabled) {
			auto W = get_W();
			bool ct = options.abft_weight_checksums_compiletime && !freivalds_enabled && !gvfa_enabled && W && W->isConst && W->data_buffer;
			if (ct) {
				INDT_2 << "/* Compile-time ABFT checksums */" << std::endl;
				if (group > 1)
					INDT_2 << "static const float b_rs_cache[" << group << "][" << tiles_per_group << "][" << K << "] = {" << std::endl;
				else
					INDT_2 << "static const float b_rs_cache[" << tiles_per_group << "][" << K << "] = {" << std::endl;
				float* wd = (float*)W->data_buffer;
				const uint64_t weight_elems = static_cast<uint64_t>(W->data_num_elem());
				uint32_t wd1 = W->data_dim.size() >= 2 ? W->data_dim[1] : 1;
				const uint32_t groups_for_cache = group > 1 ? group : 1;
				for (uint32_t gg = 0; gg < groups_for_cache; gg++) {
					if (group > 1) INDT_3 << "{" << std::endl;
					for (uint32_t t = 0; t < tiles_per_group; t++) {
						uint32_t m0 = (group > 1 ? gg * go : 0u) + mtile * t;
						uint32_t m_limit = (group > 1 ? (gg + 1) * go : (uint32_t)maps);
						uint32_t m1 = std::min(m0 + mtile, m_limit);
						std::vector<double> s(K, 0);
						for (uint32_t m = m0; m < m1 && m < (uint32_t)maps; m++) {
							uint32_t kk = 0;
							for (uint32_t c = 0; c < gi; c++) for (int ki = 0; ki < k0; ki++) {
								if (n_data_dims == 1) { uint64_t i = (uint64_t)m * gi * k0 + (uint64_t)c * k0 + (uint64_t)ki; if (i < weight_elems) s[kk++] += wd[i]; }
								else for (int kj = 0; kj < k1; kj++) { uint64_t i = ((uint64_t)m * wd1 * k0 + c * k0 + ki) * k1 + kj; if (i < weight_elems) s[kk++] += (double)wd[i]; }
							}
						}
						INDT_3 << "{";
						for (uint32_t k = 0; k < K; k++) { if (k) dst << ", "; dst << (float)s[k]; }
						dst << "}";
						if (t + 1 < tiles_per_group) dst << ",";
						dst << std::endl;
					}
					if (group > 1) {
						INDT_3 << "}" << (gg + 1 < groups_for_cache ? "," : "") << std::endl;
					}
				}
				INDT_2 << "};" << std::endl;
				auto B = get_number_of_inputs() >= 3 ? get_input_tensor(2) : nullptr;
				if (B && B->isConst && B->data_buffer) {
					float* bd = (float*)B->data_buffer;
					if (group > 1)
						INDT_2 << "static const float bias_sum_cache[" << group << "][" << tiles_per_group << "] = {" << std::endl;
					else
						INDT_2 << "static const float bias_sum_cache[" << tiles_per_group << "] = {" << std::endl;
					for (uint32_t gg = 0; gg < groups_for_cache; gg++) {
						if (group > 1) INDT_3 << "{" << std::endl;
						for (uint32_t t = 0; t < tiles_per_group; t++) {
							uint32_t m0 = (group > 1 ? gg * go : 0u) + mtile * t;
							uint32_t m_limit = (group > 1 ? (gg + 1) * go : (uint32_t)maps);
							uint32_t m1 = std::min(m0 + mtile, m_limit);
							double bs = 0;
							for (uint32_t m = m0; m < m1 && m < (uint32_t)maps; m++) bs += bd[m];
							INDT_3 << (float)bs;
							if (t + 1 < tiles_per_group) dst << ",";
							dst << std::endl;
						}
						if (group > 1) {
							INDT_3 << "}" << (gg + 1 < groups_for_cache ? "," : "") << std::endl;
						}
					}
					INDT_2 << "};" << std::endl;
				}
			} else {
				INDT_2 << "const uint32_t m_base = (uint32_t)(" << m_begin << ");" << std::endl;
				INDT_2 << "const uint32_t m_limit = (uint32_t)(" << m_end << ");" << std::endl;
				INDT_2 << "/* Runtime ABFT checksums */" << std::endl;
				INDT_2 << "float b_rs_cache[" << tiles_per_group << "][" << K << "];" << std::endl;
				INDT_2 << "float bias_sum_cache[" << tiles_per_group << "];" << std::endl;
				INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
				INDT_3 << "uint32_t m0_pre = m_base + t*MTILE;" << std::endl;
				INDT_3 << "if( m0_pre >= m_limit ) continue;" << std::endl;
				INDT_3 << "uint32_t m1_pre = MIN(m0_pre + MTILE, m_limit);" << std::endl;
				INDT_3 << "for( uint32_t kk2=0; kk2<K; kk2++ ) b_rs_cache[t][kk2] = 0;" << std::endl;
				INDT_3 << "bias_sum_cache[t] = 0;" << std::endl;
				INDT_3 << "for( uint32_t m=m0_pre; m<m1_pre; m++ ) {" << std::endl;
				if (get_number_of_inputs() >= 3) INDT_3 << "bias_sum_cache[t] += bias[m];" << std::endl;
				INDT_3 << "uint32_t kk2 = 0;" << std::endl;
				INDT_3 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
				INDT_3 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
				if (n_data_dims == 1) INDT_3 << "b_rs_cache[t][kk2++] += w[m][c0][kk0];" << std::endl;
				else { INDT_3 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl; INDT_3 << "b_rs_cache[t][kk2++] += w[m][c0][kk0][kk1];" << std::endl; INDT_3 << "}" << std::endl; }
				INDT_3 << "}" << std::endl; INDT_3 << "}" << std::endl; INDT_3 << "}" << std::endl;
				INDT_2 << "}" << std::endl;
			}
			INDT_2 << "float abft_sumC_acc[" << tiles_per_group << "];" << std::endl;
			INDT_2 << "float abft_pred_acc[" << tiles_per_group << "];" << std::endl;
			INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) { abft_sumC_acc[t] = 0.0f; abft_pred_acc[t] = 0.0f; }" << std::endl;
		}

		INDT_2 << "for( int32_t o0=0, i00=" << -pad0 << "; o0<" << out0 << "; o0++, i00+=" << s0 << " ) {" << std::endl;
		if (n_data_dims == 2) {
			INDT_3 << "for( int32_t o1=0, i10=" << -pad1 << "; o1<" << out1 << "; o1++, i10+=" << s1 << " ) {" << std::endl;
		}

		// Build im2col vector (for this output position)
		INDT_4 << "/* im2col */" << std::endl;
		INDT_4 << "uint32_t kk = 0;" << std::endl;
		INDT_4 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
		INDT_5 << "uint32_t c = c0" << (group > 1 ? " + gi*g" : "") << ";" << std::endl;
		INDT_5 << "for( int32_t k0=0; k0<" << k0 << "; k0++ ) {" << std::endl;
		INDT_5 << "int32_t i0 = i00 + k0*" << d0 << ";" << std::endl;
		if (n_data_dims == 1) {
			INDT_5 << "if( i0 >= 0 && i0 < " << in0 << " ) {" << std::endl;
			INDT_5 << "col[kk++] = x[b][c][i0];" << std::endl;
			INDT_5 << "} else {" << std::endl;
			INDT_5 << "col[kk++] = 0;" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		else {
			INDT_5 << "for( int32_t k1=0; k1<" << k1 << "; k1++ ) {" << std::endl;
			INDT_5 << "int32_t i1 = i10 + k1*" << d1 << ";" << std::endl;
			INDT_5 << "if( i0 >= 0 && i0 < " << in0 << " && i1 >= 0 && i1 < " << in1 << " ) {" << std::endl;
			INDT_5 << "col[kk++] = x[b][c][i0][i1];" << std::endl;
			INDT_5 << "} else {" << std::endl;
			INDT_5 << "col[kk++] = 0;" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		INDT_5 << "}" << std::endl;
		INDT_4 << "}" << std::endl;

		// Matmul-style: Y[m] = bias[m] + dot(col, W[m])
		INDT_4 << "/* gemm (dot-product) */" << std::endl;
		INDT_4 << "for( uint32_t m0=" << m_begin << "; m0<" << m_end << "; m0+=MTILE ) {" << std::endl;
		INDT_5 << "uint32_t m1 = MIN(m0 + MTILE, (uint32_t)(" << m_end << "));" << std::endl;

		if (options.abyzft_gemm) {
			INDT_5 << "/* AByzFT: randomized scaling (per A-row, per B-column) */" << std::endl;
			INDT_5 << "uint32_t abyzft_state = (uint32_t)(LAYER_ID ^ (uint32_t)b ^ (uint32_t)o0";
			if (n_data_dims == 2) dst << " ^ (uint32_t)o1";
			dst << " ^ (uint32_t)m0);" << std::endl;
			INDT_5 << "float abyzft_scaleA = 0.25f + 3.75f * ABYZFT_rand01(&abyzft_state);" << std::endl;
			INDT_5 << "float abyzft_scaleB[" << mtile << "];" << std::endl;
			INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) abyzft_scaleB[mi] = 0.25f + 3.75f * ABYZFT_rand01(&abyzft_state);" << std::endl;
		}

		if (checksum_enabled) {
			bool ct = options.abft_weight_checksums_compiletime && !freivalds_enabled && !gvfa_enabled && get_W() && get_W()->isConst && get_W()->data_buffer;
			if (ct) {
				INDT_5 << "uint32_t tile_idx = (m0 - (uint32_t)(" << m_begin << ")) / MTILE;" << std::endl;
				if (group > 1) {
					INDT_5 << "const float* b_rs = b_rs_cache[g][tile_idx];" << std::endl;
					INDT_5 << "float bias_sum = " << (get_number_of_inputs() >= 3 ? "bias_sum_cache[g][tile_idx]" : "0.0f") << ";" << std::endl;
				} else {
					INDT_5 << "const float* b_rs = b_rs_cache[tile_idx];" << std::endl;
					INDT_5 << "float bias_sum = " << (get_number_of_inputs() >= 3 ? "bias_sum_cache[tile_idx]" : "0.0f") << ";" << std::endl;
				}
			} else if (!freivalds_enabled && !gvfa_enabled) {
				INDT_5 << "uint32_t tile_idx = (m0 - m_base) / MTILE;" << std::endl;
				INDT_5 << "float* b_rs = b_rs_cache[tile_idx];" << std::endl;
				INDT_5 << "float bias_sum = bias_sum_cache[tile_idx];" << std::endl;
			}
			INDT_5 << "float acc_tile[" << mtile << "];" << std::endl;
		}

		INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) {" << std::endl;

		INDT_5 << y->data_type_str() << " acc = ";
		if (get_number_of_inputs() < 3)
			dst << "0;" << std::endl;
		else
			dst << "bias[m];" << std::endl;

		INDT_5 << "uint32_t k = 0;" << std::endl;
		INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
		INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
		if (n_data_dims == 1) {
			INDT_5 << "acc += col[k++] * w[m][c0][kk0];" << std::endl;
		}
		else {
			INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
			INDT_5 << "acc += col[k++] * w[m][c0][kk0][kk1];" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;

		if (options.abyzft_gemm) {
			INDT_5 << "float abyzft_scaleAB = abyzft_scaleA * abyzft_scaleB[m - m0];" << std::endl;
			INDT_5 << "float acc_scaled = acc * abyzft_scaleAB;" << std::endl;
			INDT_5 << "bool abyzft_faulted = false;" << std::endl;
			INDT_5 << "/* fault injection (on scaled result) */" << std::endl;
			INDT_5 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		} else {
			INDT_5 << "/* fault injection */" << std::endl;
			INDT_5 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		}
		if (n_data_dims == 1) {
			INDT_5 << "const uint32_t P = " << out0 << "u;" << std::endl;
			INDT_5 << "uint32_t out_idx = ((b * " << maps << "u + m) * P) + (uint32_t)o0;" << std::endl;
		}
		else {
			INDT_5 << "const uint32_t P = " << out0 << "u * " << out1 << "u;" << std::endl;
			INDT_5 << "uint32_t out_idx = ((b * " << maps << "u + m) * P) + ((uint32_t)o0 * " << out1 << "u + (uint32_t)o1);" << std::endl;
		}
		INDT_5 << "if( FAULT_MODEL==0 ) {" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( out_idx == FAULT_INDEX ) { acc_scaled += FAULT_VALUE; abyzft_faulted = true; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( out_idx == FAULT_INDEX ) { acc += FAULT_VALUE; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_5 << "/* trivial 2x2 cancelling pattern (top-left at FAULT_INDEX):" << std::endl;
		INDT_5 << " *   +e -e" << std::endl;
		INDT_5 << " *   -e +e" << std::endl;
		INDT_5 << " * in (channel x position) matrix with position stride P. */" << std::endl;
		INDT_5 << "uint32_t base_m = (FAULT_INDEX / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = FAULT_INDEX % P;" << std::endl;
		INDT_5 << "float delta = 0.0f;" << std::endl;
		if (group > 1 && go == 1 && n_data_dims == 2) {
			INDT_5 << "uint32_t base_row = base_p / " << out1 << "u;" << std::endl;
			INDT_5 << "uint32_t base_col = base_p % " << out1 << "u;" << std::endl;
			INDT_5 << "bool ok = (base_m >= m0) && (base_m < m1) && (" << out0 << "u > 1u) && (" << out1 << "u > 1u);" << std::endl;
			INDT_5 << "if( ok ) {" << std::endl;
			INDT_5 << "uint32_t row0 = MIN(base_row, " << out0 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t col0 = MIN(base_col, " << out1 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t b_off = (FAULT_INDEX / (" << maps << "u * P));" << std::endl;
			INDT_5 << "uint32_t base = ((b_off * " << maps << "u + base_m) * P) + (row0 * " << out1 << "u + col0);" << std::endl;
			INDT_5 << "if( out_idx == base ) delta = +FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base + 1u ) delta = -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base + " << out1 << "u ) delta = -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base + " << out1 << "u + 1u ) delta = +FAULT_VALUE;" << std::endl;
			INDT_5 << "}" << std::endl;
		} else {
			INDT_5 << "bool ok = true;" << std::endl;
			INDT_5 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
			INDT_5 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
			INDT_5 << "if( ok ) {" << std::endl;
			INDT_5 << "if( out_idx == FAULT_INDEX ) delta = +FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == FAULT_INDEX + P ) delta = -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +FAULT_VALUE;" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		if (options.abyzft_gemm)
			INDT_5 << "if( delta != 0.0f ) { acc_scaled += delta; abyzft_faulted = true; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_5 << "/* checkered: repeat the trivial 2x2 cancelling pattern FAULT_N times at different positions. */" << std::endl;
		INDT_5 << "float delta = 0.0f;" << std::endl;
		INDT_5 << "for( uint32_t r=0; r<FAULT_N; r++ ) {" << std::endl;
		INDT_5 << "uint32_t base = FAULT_INDEX + r * FAULT_STRIDE;" << std::endl;
		INDT_5 << "uint32_t base_m = (base / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = base % P;" << std::endl;
		if (group > 1 && go == 1 && n_data_dims == 2) {
			INDT_5 << "uint32_t base_row = base_p / " << out1 << "u;" << std::endl;
			INDT_5 << "uint32_t base_col = base_p % " << out1 << "u;" << std::endl;
			INDT_5 << "bool ok = (base_m >= m0) && (base_m < m1) && (" << out0 << "u > 1u) && (" << out1 << "u > 1u);" << std::endl;
			INDT_5 << "if( !ok ) continue;" << std::endl;
			INDT_5 << "uint32_t row0 = MIN(base_row, " << out0 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t col0 = MIN(base_col, " << out1 << "u - 2u);" << std::endl;
			INDT_5 << "uint32_t b_off = (base / (" << maps << "u * P));" << std::endl;
			INDT_5 << "uint32_t base2 = ((b_off * " << maps << "u + base_m) * P) + (row0 * " << out1 << "u + col0);" << std::endl;
			INDT_5 << "if( out_idx == base2 ) delta += +FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base2 + 1u ) delta += -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base2 + " << out1 << "u ) delta += -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base2 + " << out1 << "u + 1u ) delta += +FAULT_VALUE;" << std::endl;
		} else {
			INDT_5 << "bool ok = true;" << std::endl;
			INDT_5 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
			INDT_5 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
			INDT_5 << "if( !ok ) continue;" << std::endl;
			INDT_5 << "if( out_idx == base ) delta += +FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base + 1u ) delta += -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base + P ) delta += -FAULT_VALUE;" << std::endl;
			INDT_5 << "else if( out_idx == base + P + 1u ) delta += +FAULT_VALUE;" << std::endl;
		}
		INDT_5 << "}" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( delta != 0.0f ) { acc_scaled += delta; abyzft_faulted = true; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;

		if (options.abyzft_gemm)
			INDT_5 << "if( abyzft_faulted && abyzft_scaleAB != 0.0f ) acc = acc_scaled / abyzft_scaleAB;" << std::endl;

		if (n_data_dims == 1)
			INDT_5 << "y[b][m][o0] = acc;" << std::endl;
		else
			INDT_5 << "y[b][m][o0][o1] = acc;" << std::endl;

		if (checksum_enabled)
			INDT_5 << "acc_tile[m-m0] = acc;" << std::endl;

		INDT_5 << "} /* m */" << std::endl;

		if (checksum_enabled) {
			if (freivalds_enabled || gvfa_enabled) {
				INDT_5 << "/* Randomized verify (Freivalds/GVFA): r^T C_tile == (A^T (B_tile r)) + bias */" << std::endl;
				INDT_5 << "for( uint32_t chk=0; chk<" << randomized_checks << "u; chk++ ) {" << std::endl;
				INDT_5 << "uint32_t freivalds_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ (uint32_t)b ^ (uint32_t)o0";
				if (n_data_dims == 2) dst << " ^ (uint32_t)o1";
				dst << " ^ (uint32_t)m0 ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
				if (freivalds_enabled)
					INDT_5 << "uint8_t r_mask[" << mtile << "];" << std::endl;
				else
					INDT_5 << "float r_vec[" << mtile << "];" << std::endl;
				if (freivalds_enabled) {
					INDT_5 << "uint32_t r_any = 0;" << std::endl;
					INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) { uint32_t bit = ABYZFT_randbit(&freivalds_state); r_mask[mi] = (uint8_t)bit; r_any |= bit; }" << std::endl;
					INDT_5 << "if( r_any == 0 ) r_mask[0] = 1u;" << std::endl;
				} else {
					INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) r_vec[mi] = ABYZFT_randn(&freivalds_state);" << std::endl;
				}
				INDT_5 << "float b_rs[" << K << "];" << std::endl;
				INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) b_rs[kk2] = 0;" << std::endl;
				INDT_5 << "float bias_sum = 0;" << std::endl;
				INDT_5 << "float sumC = 0;" << std::endl;
				INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) {" << std::endl;
				if (freivalds_enabled) {
					INDT_5 << "if( !r_mask[m-m0] ) continue;" << std::endl;
					if (get_number_of_inputs() >= 3)
						INDT_5 << "bias_sum += bias[m];" << std::endl;
					INDT_5 << "sumC += acc_tile[m-m0];" << std::endl;
					INDT_5 << "uint32_t kk2 = 0;" << std::endl;
					INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
					INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
					if (n_data_dims == 1) {
						INDT_5 << "b_rs[kk2++] += w[m][c0][kk0];" << std::endl;
					}
					else {
						INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
						INDT_5 << "b_rs[kk2++] += w[m][c0][kk0][kk1];" << std::endl;
						INDT_5 << "}" << std::endl;
					}
					INDT_5 << "}" << std::endl;
					INDT_5 << "}" << std::endl;
				} else {
					if (get_number_of_inputs() >= 3)
						INDT_5 << "bias_sum += bias[m] * r_vec[m-m0];" << std::endl;
					INDT_5 << "sumC += acc_tile[m-m0] * r_vec[m-m0];" << std::endl;
					INDT_5 << "uint32_t kk2 = 0;" << std::endl;
					INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
					INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
					if (n_data_dims == 1) {
						INDT_5 << "b_rs[kk2++] += w[m][c0][kk0] * r_vec[m-m0];" << std::endl;
					}
					else {
						INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
						INDT_5 << "b_rs[kk2++] += w[m][c0][kk0][kk1] * r_vec[m-m0];" << std::endl;
						INDT_5 << "}" << std::endl;
					}
					INDT_5 << "}" << std::endl;
					INDT_5 << "}" << std::endl;
				}
				INDT_5 << "}" << std::endl;
				INDT_5 << "float pred = bias_sum;" << std::endl;
				INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += col[kk2] * b_rs[kk2];" << std::endl;
				INDT_5 << "float diff = fabsf(sumC - pred);" << std::endl;
				INDT_5 << "float tol = " << (options.abyzft_gemm ? (options.abft_eps * 0.1f) : options.abft_eps) << "f * (fabsf(pred) + 1.0f);" << std::endl;
				INDT_5 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
				INDT_5 << "}" << std::endl;
			}
			else {
				INDT_5 << "/* Verify checksum */" << std::endl;
				INDT_5 << "float sumC = 0;" << std::endl;
				INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) sumC += acc_tile[mi];" << std::endl;
				INDT_5 << "float pred = bias_sum;" << std::endl;
				INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += col[kk2] * b_rs[kk2];" << std::endl;
				INDT_5 << "abft_sumC_acc[tile_idx] += sumC;" << std::endl;
				INDT_5 << "abft_pred_acc[tile_idx] += pred;" << std::endl;
			}
		}

		INDT_4 << "} /* m0 */" << std::endl;

		if (n_data_dims == 2)
			INDT_3 << "} /* o1 */" << std::endl;
		INDT_2 << "} /* o0 */" << std::endl;
		if (checksum_enabled && !randomized_enabled) {
			INDT_2 << "for( uint32_t t=0; t<" << tiles_per_group << "; t++ ) {" << std::endl;
			INDT_3 << "float pred_acc = abft_pred_acc[t];" << std::endl;
			INDT_3 << "float diff = fabsf(abft_sumC_acc[t] - pred_acc);" << std::endl;
			INDT_3 << "float tol = " << (options.abyzft_gemm ? (options.abft_eps * 0.1f) : options.abft_eps) << "f * (fabsf(pred_acc) + 1.0f);" << std::endl;
			INDT_3 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
			INDT_2 << "}" << std::endl;
		}

		if (group > 1)
			INDT_1 << "} /* g */" << std::endl;
		INDT_1 << "} /* b */" << std::endl;
	}

	virtual void print_output_cell_init(std::ostream& dst, const std::string& y_idx) const override
	{
		INDT_3 << "y" << y_idx << " = ";
		if (get_number_of_inputs() < 3) // bias is the 3rd input, optional
			dst << "0;" << std::endl;
		else
			dst << "bias[m];" << std::endl;
	};
	virtual void print_output_cell_calc(
	    std::ostream& dst,
	    const std::string& x_idx,
	    const std::string& w_idx,
	    const std::string& y_idx) const override
	{
		INDT_4 << "y" << y_idx << " += x" << x_idx << " * w" << w_idx << ";" << std::endl;
	}
	virtual void print_output_cell_finalize(std::ostream& dst, const std::string& y_idx) const override
	{
	}
	virtual void print(std::ostream& dst) const override
	{
		print_header_info_comment(dst);
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || options.freivalds_gemm || options.gvfa_gemm;
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
		if (get_number_of_inputs() == 3) {
			name_input(2, "bias");
		}

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
