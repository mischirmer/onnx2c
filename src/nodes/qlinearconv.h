/* This file is part of onnx2c.
 *
 * QLinearConv
 *
 * Integer convolution with explicit quantization parameters (QOperator form).
 *
 * Implemented for rank-3 (NCW) and rank-4 (NCHW) tensors by lowering to
 * im2col + dot-product with int32 accumulators.
 */

#pragma once

#include "node.h"
#include "../options.h"

namespace toC {

class QLinearConv : public Node {
	public:
	QLinearConv()
	{
		op_name = "QLinearConv";
		auto_pad = "NOTSET";
		group = 1;
	}

	// Attributes (mirrors SpatialFilter attributes)
	std::vector<int64_t> kernel_shape;
	std::string auto_pad;
	std::vector<int64_t> dilations;
	int group;
	std::vector<int64_t> pads;
	std::vector<int64_t> strides;

	const Tensor* get_X(void) const { return get_input_tensor(0); }
	const Tensor* get_W(void) const { return get_input_tensor(3); }
	const Tensor* get_Y(void) const { return get_output_tensor(0); }
	uint32_t get_numDataDim(void) const { return get_X()->rank() - 2; }

	void name_scalar_input(unsigned input_no, std::string name)
	{
		name_input(input_no, name);
		const Tensor* t = get_input_tensor(input_no);
		if (!(t->data_dim.size() == 0 || (t->data_dim.size() == 1 && t->data_dim[0] == 1))) {
			ERROR(name << " must be scalar");
		}
	}

	void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			if (a.name() == "auto_pad")
				auto_pad = parse_attribute_string(a);
			else if (a.name() == "dilations")
				dilations = parse_attribute_ints(a);
			else if (a.name() == "group")
				group = parse_attribute_int(a);
			else if (a.name() == "kernel_shape")
				kernel_shape = parse_attribute_ints(a);
			else if (a.name() == "pads")
				pads = parse_attribute_ints(a);
			else if (a.name() == "strides")
				strides = parse_attribute_ints(a);
		}
	}

	void resolve_strides(void)
	{
		if (strides.size() == 0)
			for (unsigned i = 0; i < get_numDataDim(); i++)
				strides.push_back(1);
		if (get_numDataDim() != strides.size())
			ERROR("Dimension of the stride do not match data dimensions");
		for (uint64_t s : strides)
			if (s == 0)
				ERROR("Stride of 0");
	}

	void resolve_kernel_shape(void)
	{
		if (kernel_shape.size() == 0) {
			for (unsigned i = 2; i < get_W()->rank(); i++)
				kernel_shape.push_back(get_W()->data_dim[i]);
		}
	}

	void resolve_dilations(void)
	{
		if (dilations.size() == 0)
			for (unsigned i = 0; i < get_numDataDim(); i++)
				dilations.push_back(1);
	}

	void resolve_pads(void)
	{
		unsigned num_data_dim = get_numDataDim();
		if (pads.size() == 0) {
			pads.resize(num_data_dim * 2);
			for (unsigned i = 0; i < num_data_dim; i++) {
				if (auto_pad == "VALID" || auto_pad == "NOTSET") {
					pads[i] = 0;
					pads[i + num_data_dim] = 0;
				}
				else {
					pads[i] = kernel_shape[i] / 2;
					pads[i + num_data_dim] = kernel_shape[i] / 2;
				}
			}
		}
	}

	std::vector<int> resolve_output_size(void) const
	{
		std::vector<int> rv;
		unsigned num_data_dim = get_numDataDim();
		rv.push_back(get_X()->data_dim[0]); // batch
		rv.push_back(get_W()->data_dim[0]); // maps

		for (unsigned dim = 0, xdim = 2; dim < num_data_dim; dim++, xdim++) {
			int outdim;
			int filter_size = kernel_shape[dim];
			filter_size += (kernel_shape[dim] - 1) * (dilations[dim] - 1);

			if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
				outdim = get_X()->data_dim[xdim];
			}
			else if (auto_pad == "NOTSET" || auto_pad == "VALID") {
				int input_size = get_X()->data_dim[xdim] + pads[dim] + pads[dim + num_data_dim];
				int last_out = input_size - filter_size;
				outdim = last_out / strides[dim] + 1;
			}
			else {
				ERROR("Invalid option for auto_pad attribute");
			}
			rv.push_back(outdim);
		}
		return rv;
	}

	void print_header_info_comment(std::ostream& dst) const
	{
		INDT_1 << "/* " << op_name << std::endl;
		INDT_1 << " * node_id: " << node_id << std::endl;
		INDT_1 << " *" << std::endl;
		INDT_1 << " * auto_pad: " << auto_pad << std::endl;
		INDT_1 << " * dilations: ";
		for (int d : dilations)
			dst << d << " ";
		dst << std::endl;
		INDT_1 << " * group: " << group << std::endl;
		INDT_1 << " * kernel_shape: ";
		for (int k : kernel_shape)
			dst << k << " ";
		dst << std::endl;
		INDT_1 << " * pads: ";
		for (int p : pads)
			dst << p << " ";
		dst << std::endl;
		INDT_1 << " * strides: ";
		for (int s : strides)
			dst << s << " ";
		dst << std::endl;
		INDT_1 << " */" << std::endl;
	}

	void resolve(void) override
	{
		// Inputs:
		// 0 X
		// 1 x_scale
		// 2 x_zero_point
		// 3 W
		// 4 w_scale
		// 5 w_zero_point
		// 6 y_scale
		// 7 y_zero_point
		// 8 B (optional)
		name_input(0, "x");
		name_scalar_input(1, "x_scale");
		name_scalar_input(2, "x_zero_point");
		name_input(3, "w");
		name_input(4, "w_scale");
		name_input(5, "w_zero_point");
		name_scalar_input(6, "y_scale");
		name_scalar_input(7, "y_zero_point");
		if (get_number_of_inputs() >= 9)
			name_input(8, "bias");

		if (get_X()->rank() < 3 || get_X()->rank() > 4)
			ERROR("Unimplemented: QLinearConv for input rank " << get_X()->rank());

		resolve_strides();
		resolve_dilations();
		resolve_kernel_shape();
		resolve_pads();

		for (int d : dilations)
			if (d != 1)
				ERROR("Unimplemented: QLinearConv: dilations other than 1");

		const Tensor* y_zero_point = get_input_tensor(7);
		Tensor* y = new Tensor;
		y->data_dim = resolve_output_size();
		y->data_type = y_zero_point->data_type;
		register_output(y, "y");
	}

	void print(std::ostream& dst) const override
	{
		print_header_info_comment(dst);

		const uint32_t n_data_dims = get_numDataDim();
		if (n_data_dims != 1 && n_data_dims != 2) {
			ERROR("Unimplemented: QLinearConv only supports 1D/2D conv at the moment");
		}

		const auto* x = get_X();
		const auto* w_scale_t = get_input_tensor(4);
		const auto* w_zero_t = get_input_tensor(5);
		const auto* y = get_Y();

		const bool w_scale_scalar = w_scale_t->is_scalar() || (w_scale_t->rank() == 1 && w_scale_t->data_dim[0] == 1);
		const bool w_zero_scalar = w_zero_t->is_scalar() || (w_zero_t->rank() == 1 && w_zero_t->data_dim[0] == 1);
		const std::string w_scale_idx = w_scale_scalar ? "[0]" : "[m]";
		const std::string w_zero_idx = w_zero_scalar ? "[0]" : "[m]";

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
		const uint32_t K = (n_data_dims == 1) ? (gi * k0) : (gi * k0 * k1);

		INDT_1 << "/* QLinearConv (im2col + dot-product) */" << std::endl;
		INDT_1 << "const uint32_t LAYER_ID = " << node_id << ";" << std::endl;
		INDT_1 << "const uint32_t K = " << K << ";" << std::endl;
		INDT_1 << "int32_t col[" << K << "];" << std::endl;

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

		INDT_4 << "/* im2col: A_k = (x - x_zero_point) */" << std::endl;
		INDT_4 << "uint32_t kk = 0;" << std::endl;
		INDT_4 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
		INDT_5 << "uint32_t c = c0" << (group > 1 ? " + gi*g" : "") << ";" << std::endl;
		INDT_5 << "for( int32_t k0=0; k0<" << k0 << "; k0++ ) {" << std::endl;
		INDT_5 << "int32_t i0 = i00 + k0*" << d0 << ";" << std::endl;
		if (n_data_dims == 1) {
			INDT_5 << "if( i0 >= 0 && i0 < " << in0 << " ) {" << std::endl;
			INDT_5 << "col[kk++] = ((int32_t)x[b][c][i0] - (int32_t)x_zero_point[0]);" << std::endl;
			INDT_5 << "} else {" << std::endl;
			INDT_5 << "col[kk++] = 0;" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		else {
			INDT_5 << "for( int32_t k1=0; k1<" << k1 << "; k1++ ) {" << std::endl;
			INDT_5 << "int32_t i1 = i10 + k1*" << d1 << ";" << std::endl;
			INDT_5 << "if( i0 >= 0 && i0 < " << in0 << " && i1 >= 0 && i1 < " << in1 << " ) {" << std::endl;
			INDT_5 << "col[kk++] = ((int32_t)x[b][c][i0][i1] - (int32_t)x_zero_point[0]);" << std::endl;
			INDT_5 << "} else {" << std::endl;
			INDT_5 << "col[kk++] = 0;" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		INDT_5 << "}" << std::endl;
		INDT_4 << "}" << std::endl;

		INDT_4 << "/* gemm (dot-product): acc32 = bias + sum_k (A_k * B_mk) */" << std::endl;
		const uint32_t mtile = options.abft_mtile ? options.abft_mtile : 8;
		INDT_4 << "const uint32_t MTILE = " << mtile << ";" << std::endl;

		std::string m_begin = (group > 1) ? "go*g" : "0";
		std::string m_end = (group > 1) ? "go*(g+1)" : std::to_string(maps);

		INDT_4 << "for( uint32_t m0=" << m_begin << "; m0<" << m_end << "; m0+=MTILE ) {" << std::endl;
		INDT_5 << "uint32_t m1 = MIN(m0 + MTILE, (uint32_t)(" << m_end << "));" << std::endl;

		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm;
		if (options.abyzft_gemm) {
			INDT_5 << "/* AByzFT: randomized scaling (per A-row, per B-column) */" << std::endl;
			INDT_5 << "uint32_t abyzft_state = (uint32_t)(LAYER_ID ^ (uint32_t)b ^ (uint32_t)o0";
			if (n_data_dims == 2) dst << " ^ (uint32_t)o1";
			dst << " ^ (uint32_t)m0);" << std::endl;
			INDT_5 << "float abyzft_scaleA = 0.5f + ABYZFT_rand01(&abyzft_state);" << std::endl;
			INDT_5 << "float abyzft_scaleB[" << mtile << "];" << std::endl;
			INDT_5 << "for( uint32_t mi=0; mi<(m1-m0); mi++ ) abyzft_scaleB[mi] = 0.5f + ABYZFT_rand01(&abyzft_state);" << std::endl;
		}

		if (checksum_enabled) {
			INDT_5 << "/* ABFT (integer domain): build row-checksum of B over output-channel tile */" << std::endl;
			INDT_5 << "int32_t b_rs[" << K << "];" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) b_rs[kk2] = 0;" << std::endl;
			INDT_5 << "int64_t bias_sum = 0;" << std::endl;
			INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) {" << std::endl;
			if (get_number_of_inputs() >= 9)
				INDT_5 << "bias_sum += (int64_t)bias[m];" << std::endl;
			INDT_5 << "uint32_t kk2 = 0;" << std::endl;
			INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
			INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
			if (n_data_dims == 1) {
				INDT_5 << "b_rs[kk2++] += ((int32_t)w[m][c0][kk0] - (int32_t)w_zero_point" << w_zero_idx << ");" << std::endl;
			}
			else {
				INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
				INDT_5 << "b_rs[kk2++] += ((int32_t)w[m][c0][kk0][kk1] - (int32_t)w_zero_point" << w_zero_idx << ");" << std::endl;
				INDT_5 << "}" << std::endl;
			}
			INDT_5 << "}" << std::endl;
			INDT_5 << "}" << std::endl;
			INDT_5 << "} /* m checksum */" << std::endl;
			INDT_5 << "int64_t sumC = 0;" << std::endl;
		}

		INDT_5 << "for( uint32_t m=m0; m<m1; m++ ) {" << std::endl;
		INDT_5 << "int32_t acc32 = " << (get_number_of_inputs() >= 9 ? "bias[m];" : "0;") << std::endl;

		INDT_5 << "uint32_t k = 0;" << std::endl;
		INDT_5 << "for( uint32_t c0=0; c0<" << gi << "; c0++ ) {" << std::endl;
		INDT_5 << "for( int32_t kk0=0; kk0<" << k0 << "; kk0++ ) {" << std::endl;
		if (n_data_dims == 1) {
			INDT_5 << "acc32 += col[k++] * (((int32_t)w[m][c0][kk0]) - (int32_t)w_zero_point" << w_zero_idx << ");" << std::endl;
		}
		else {
			INDT_5 << "for( int32_t kk1=0; kk1<" << k1 << "; kk1++ ) {" << std::endl;
			INDT_5 << "acc32 += col[k++] * (((int32_t)w[m][c0][kk0][kk1]) - (int32_t)w_zero_point" << w_zero_idx << ");" << std::endl;
			INDT_5 << "}" << std::endl;
		}
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;

		if (options.abyzft_gemm) {
			INDT_5 << "float abyzft_scaleAB = abyzft_scaleA * abyzft_scaleB[m - m0];" << std::endl;
			INDT_5 << "double acc_scaled = (double)acc32 * (double)abyzft_scaleAB;" << std::endl;
			INDT_5 << "bool abyzft_faulted = false;" << std::endl;
			INDT_5 << "/* fault injection (on scaled accumulator) */" << std::endl;
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
		// Interpret FAULT_VALUE as a delta in real (fp32) domain and convert it to an
		// equivalent delta on the int32 accumulator:  delta_acc32 ~= FAULT_VALUE / (x_scale*w_scale).
		INDT_5 << "double eff_scale = (double)x_scale[0] * (double)w_scale" << w_scale_idx << ";" << std::endl;
		INDT_5 << "int32_t fault_delta = 0;" << std::endl;
		INDT_5 << "if( eff_scale != 0.0 ) fault_delta = (int32_t) llround((double)FAULT_VALUE / eff_scale);" << std::endl;
		INDT_5 << "if( FAULT_MODEL==0 ) {" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( out_idx == FAULT_INDEX ) { acc_scaled += (double)fault_delta; abyzft_faulted = true; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( out_idx == FAULT_INDEX ) { acc32 += fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "} else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_5 << "bool trivial_ok = (" << group << "==1);" << std::endl;
		INDT_5 << "uint32_t base_m = (FAULT_INDEX / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = FAULT_INDEX % P;" << std::endl;
		INDT_5 << "trivial_ok = trivial_ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
		INDT_5 << "trivial_ok = trivial_ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
		INDT_5 << "int32_t delta = 0;" << std::endl;
		INDT_5 << "if( trivial_ok ) {" << std::endl;
		INDT_5 << "if( out_idx == FAULT_INDEX ) delta = +fault_delta;" << std::endl;
		INDT_5 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -fault_delta;" << std::endl;
		INDT_5 << "else if( out_idx == FAULT_INDEX + P ) delta = -fault_delta;" << std::endl;
		INDT_5 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +fault_delta;" << std::endl;
		INDT_5 << "}" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( delta != 0 ) { acc_scaled += (double)delta; abyzft_faulted = true; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( delta != 0 ) { acc32 += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "} else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_5 << "int32_t delta = 0;" << std::endl;
		INDT_5 << "for( uint32_t r=0; r<FAULT_N; r++ ) {" << std::endl;
		INDT_5 << "uint32_t base = FAULT_INDEX + r * FAULT_STRIDE;" << std::endl;
		INDT_5 << "uint32_t base_m = (base / P) % " << maps << "u;" << std::endl;
		INDT_5 << "uint32_t base_p = base % P;" << std::endl;
		INDT_5 << "bool ok = (" << group << "==1);" << std::endl;
		INDT_5 << "ok = ok && (base_m + 1u < " << maps << "u) && (base_p + 1u < P);" << std::endl;
		INDT_5 << "ok = ok && (base_m >= m0) && (base_m + 1u < m1);" << std::endl;
		INDT_5 << "if( !ok ) continue;" << std::endl;
		INDT_5 << "if( out_idx == base ) delta += +fault_delta;" << std::endl;
		INDT_5 << "else if( out_idx == base + 1u ) delta += -fault_delta;" << std::endl;
		INDT_5 << "else if( out_idx == base + P ) delta += -fault_delta;" << std::endl;
		INDT_5 << "else if( out_idx == base + P + 1u ) delta += +fault_delta;" << std::endl;
		INDT_5 << "}" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( delta != 0 ) { acc_scaled += (double)delta; abyzft_faulted = true; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( delta != 0 ) { acc32 += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_5 << "}" << std::endl;
		INDT_5 << "}" << std::endl;

		if (options.abyzft_gemm) {
			// Avoid perturbing fault-free results: only map back to int32 when we actually
			// injected a fault into this output element. Otherwise keep the exact acc32.
			INDT_5 << "if( abyzft_faulted && abyzft_scaleAB != 0.0f ) acc32 = (int32_t) llround(acc_scaled / (double)abyzft_scaleAB);" << std::endl;
		}

		if (checksum_enabled) {
			INDT_5 << "sumC += (int64_t)acc32;" << std::endl;
		}

		std::string float_dtype = get_input_tensor(1)->data_type_str();
		INDT_5 << float_dtype << " scale = (" << float_dtype << ") (x_scale[0] * w_scale" << w_scale_idx << ") / y_scale[0];" << std::endl;
		INDT_5 << "double scaled = ((double) acc32) * (double) scale;" << std::endl;
		INDT_5 << "scaled = scaled + (double) y_zero_point[0];" << std::endl;
		INDT_5 << "int t = (int) llround(scaled);" << std::endl;
		auto [lower, upper] = y->get_type_bounds();
		INDT_5 << "t = MIN(MAX(t, " << lower << "), " << upper << ");" << std::endl;
		if (n_data_dims == 1)
			INDT_5 << "y[b][m][o0] = (" << y->data_type_str() << ") t;" << std::endl;
		else
			INDT_5 << "y[b][m][o0][o1] = (" << y->data_type_str() << ") t;" << std::endl;

		INDT_5 << "} /* m */" << std::endl;

		if (checksum_enabled) {
			INDT_5 << "/* ABFT verify (integer domain): sum(C_tile) == A * (B_tile 1) */" << std::endl;
			INDT_5 << "int64_t pred = bias_sum;" << std::endl;
			INDT_5 << "for( uint32_t kk2=0; kk2<K; kk2++ ) pred += (int64_t)col[kk2] * (int64_t)b_rs[kk2];" << std::endl;
			INDT_5 << "if( sumC != pred ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
		}

		INDT_4 << "} /* m0 */" << std::endl;

		if (n_data_dims == 2)
			INDT_3 << "} /* o1 */" << std::endl;
		INDT_2 << "} /* o0 */" << std::endl;

		if (group > 1)
			INDT_1 << "} /* g */" << std::endl;
		INDT_1 << "} /* b */" << std::endl;
	}
};

} // namespace toC
