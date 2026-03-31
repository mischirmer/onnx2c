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
		if (group > 1) {
			INDT_4 << "for( uint32_t m=go*g; m<go*(g+1); m++ ) {" << std::endl;
		} else {
			INDT_4 << "for( uint32_t m=0; m<" << maps << "; m++ ) {" << std::endl;
		}

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

		if (n_data_dims == 1)
			INDT_5 << "y[b][m][o0] = acc;" << std::endl;
		else
			INDT_5 << "y[b][m][o0][o1] = acc;" << std::endl;
		INDT_4 << "} /* m */" << std::endl;

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
