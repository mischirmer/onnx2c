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
		INDT_1 << "const uint32_t LAYER_ID = " << node_id << ";" << std::endl;
		INDT_1 << "const uint32_t K = " << K << ";" << std::endl;
		INDT_1 << x->data_type_str() << " col[" << K << "];" << std::endl;

		INDT_1 << "for( uint32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		if (group > 1) {
			INDT_1 << "uint32_t go = " << go << "; // output group size, i.e. maps/group" << std::endl;
			INDT_1 << "uint32_t gi = " << gi << "; // input group size, i.e. channels/group" << std::endl;
			INDT_1 << "for( uint32_t g=0; g<" << group << "; g++ ) {" << std::endl;
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
		const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 8;
		INDT_4 << "const uint32_t MTILE = " << mtile << ";" << std::endl;

		std::string m_begin = (group > 1) ? "go*g" : "0";
		std::string m_end = (group > 1) ? "go*(g+1)" : std::to_string(maps);

		INDT_4 << "for( uint32_t m0=" << m_begin << "; m0<" << m_end << "; m0+=MTILE ) {" << std::endl;
		INDT_5 << "uint32_t m1 = MIN(m0 + MTILE, (uint32_t)(" << m_end << "));" << std::endl;

		if (options.abft_gemm) {
			INDT_5 << "/* ABFT: build row-checksum of B (Kx1) over output-channel tile */" << std::endl;
			INDT_5 << "float b_rs[" << K << "];" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) b_rs[kk2] = 0;" << std::endl;
			INDT_5 << "float bias_sum = 0;" << std::endl;
			INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) {" << std::endl;
			if (get_number_of_inputs() >= 3)
				INDT_5 << "bias_sum += bias[m];" << std::endl;

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
			INDT_5 << "} /* m checksum */" << std::endl;
			INDT_5 << "float sumC = 0;" << std::endl;
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

		INDT_5 << "/* fault injection */" << std::endl;
		INDT_5 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		if (n_data_dims == 1) {
			INDT_5 << "const uint32_t P = " << out0 << "u;" << std::endl;
			INDT_5 << "uint32_t out_idx = ((b * " << maps << "u + m) * P) + (uint32_t)o0;" << std::endl;
		}
		else {
			INDT_5 << "const uint32_t P = " << out0 << "u * " << out1 << "u;" << std::endl;
			INDT_5 << "uint32_t out_idx = ((b * " << maps << "u + m) * P) + ((uint32_t)o0 * " << out1 << "u + (uint32_t)o1);" << std::endl;
		}
		INDT_5 << "if( FAULT_MODEL==0 ) {" << std::endl;
		INDT_5 << "if( out_idx == FAULT_INDEX ) { acc += FAULT_VALUE; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_5 << "/* trivial 2x2 cancelling pattern (top-left at FAULT_INDEX):" << std::endl;
		INDT_5 << " *   +e -e" << std::endl;
		INDT_5 << " *   -e +e" << std::endl;
		INDT_5 << " * in (channel x position) matrix with position stride P. */" << std::endl;
		INDT_5 << "/* Only meaningful when the checksum sums across >=2 channels (i.e. group==1 and tile has >=2 m). */" << std::endl;
		INDT_5 << "bool trivial_ok = (" << group << "==1);" << std::endl;
		INDT_5 << "uint32_t base_m = (FAULT_INDEX / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = FAULT_INDEX % P;" << std::endl;
		INDT_5 << "trivial_ok = trivial_ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
		INDT_5 << "trivial_ok = trivial_ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
		INDT_5 << "float delta = 0.0f;" << std::endl;
		INDT_5 << "if( trivial_ok ) {" << std::endl;
		INDT_5 << "if( out_idx == FAULT_INDEX ) delta = +FAULT_VALUE;" << std::endl;
		INDT_5 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -FAULT_VALUE;" << std::endl;
		INDT_5 << "else if( out_idx == FAULT_INDEX + P ) delta = -FAULT_VALUE;" << std::endl;
		INDT_5 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +FAULT_VALUE;" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_5 << "/* checkered: repeat the trivial 2x2 cancelling pattern FAULT_N times at different positions. */" << std::endl;
		INDT_5 << "float delta = 0.0f;" << std::endl;
		INDT_5 << "for( uint32_t r=0; r<FAULT_N; r++ ) {" << std::endl;
		INDT_5 << "uint32_t base = FAULT_INDEX + r * FAULT_STRIDE;" << std::endl;
		INDT_5 << "uint32_t base_m = (base / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = base % P;" << std::endl;
		INDT_5 << "bool ok = (" << group << "==1);" << std::endl;
		INDT_5 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
		INDT_5 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
		INDT_5 << "if( !ok ) continue;" << std::endl;
		INDT_5 << "if( out_idx == base ) delta += +FAULT_VALUE;" << std::endl;
		INDT_5 << "else if( out_idx == base + 1u ) delta += -FAULT_VALUE;" << std::endl;
		INDT_5 << "else if( out_idx == base + P ) delta += -FAULT_VALUE;" << std::endl;
		INDT_5 << "else if( out_idx == base + P + 1u ) delta += +FAULT_VALUE;" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;

		if (n_data_dims == 1)
			INDT_5 << "y[b][m][o0] = acc;" << std::endl;
		else
			INDT_5 << "y[b][m][o0][o1] = acc;" << std::endl;

		if (options.abft_gemm) {
			INDT_5 << "sumC += acc;" << std::endl;
		}

		INDT_5 << "} /* m */" << std::endl;

		if (options.abft_gemm) {
			INDT_5 << "/* ABFT verify: sum(C_tile) == (1^T A_tile) * (B_tile 1) */" << std::endl;
			INDT_5 << "float pred = bias_sum;" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += col[kk2] * b_rs[kk2];" << std::endl;
			INDT_5 << "float diff = fabsf(sumC - pred);" << std::endl;
			INDT_5 << "float tol = " << options.abft_eps << "f * (fabsf(pred) + 1.0f);" << std::endl;
			INDT_5 << "if( diff > tol ) {" << std::endl;
			INDT_5 << "TAMPERING_DETECTED = true;" << std::endl;
			INDT_5 << "TAMPERING_DETECTIONS++;" << std::endl;
			INDT_5 << "/* ABFT failure (tile m0..m1-1 at this output position) */" << std::endl;
			INDT_5 << "}" << std::endl;
		}

		INDT_4 << "} /* m0 */" << std::endl;

		if (n_data_dims == 2)
			INDT_3 << "} /* o1 */" << std::endl;
		INDT_2 << "} /* o0 */" << std::endl;

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
		if (options.conv_im2col) {
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
