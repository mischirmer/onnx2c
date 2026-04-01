/* This file is part of onnx2c.
 *
 * QGemm
 *
 * Quantized GEMM (QOperator form).
 *
 * This implementation targets the common post-training quantization pattern:
 *   inputs: A, a_scale, a_zero_point, B, b_scale, b_zero_point, C, y_scale, y_zero_point
 * with scalar scales/zero-points for A/B/Y and an int32 bias C (broadcastable).
 */

#pragma once

#include "node.h"
#include "../options.h"
#include <cmath>

namespace toC {

class QGemm : public Node {
	public:
	QGemm()
	{
		op_name = "QGemm";
		alpha = 1.0f;
		transA = 0;
		transB = 0;
	}

	float alpha;
	int transA;
	int transB;

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
			if (a.name() == "alpha")
				alpha = parse_attribute_float(a);
			else if (a.name() == "transA")
				transA = parse_attribute_int(a);
			else if (a.name() == "transB")
				transB = parse_attribute_int(a);
		}
	}

	void resolve(void) override
	{
		// Inputs:
		// 0 A
		// 1 a_scale
		// 2 a_zero_point
		// 3 B
		// 4 b_scale
		// 5 b_zero_point
		// 6 C (bias, int32, broadcastable) - required in our current models
		// 7 y_scale
		// 8 y_zero_point

		name_input(0, "A");
		name_scalar_input(1, "a_scale");
		name_scalar_input(2, "a_zero_point");
		name_input(3, "B");
		name_scalar_input(4, "b_scale");
		name_scalar_input(5, "b_zero_point");
		name_input(6, "C");
		name_scalar_input(7, "y_scale");
		name_scalar_input(8, "y_zero_point");

		if (transA != 0)
			ERROR("Unimplemented: QGemm transA");
		if (alpha != 1.0f)
			ERROR("Unimplemented: QGemm alpha != 1");

		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(3);
		const Tensor* y_zero_point = get_input_tensor(8);

		if (A->rank() != 2 || B->rank() != 2)
			ERROR("Unimplemented: QGemm only supports rank-2 matrices");

		int M = A->data_dim[0];
		int K = A->data_dim[1];
		int N = transB ? B->data_dim[0] : B->data_dim[1];
		int Kb = transB ? B->data_dim[1] : B->data_dim[0];
		if (Kb != K)
			ERROR("Reduction dimension mismatch in QGemm");

		Tensor* Y = new Tensor;
		Y->data_dim = {M, N};
		Y->data_type = y_zero_point->data_type;
		register_output(Y, "Y");
	}

	void print(std::ostream& dst) const override
	{
		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(3);
		const Tensor* C = get_input_tensor(6);
		const Tensor* Y = get_output_tensor(0);

		int M = A->data_dim[0];
		int K = A->data_dim[1];
		int N = transB ? B->data_dim[0] : B->data_dim[1];

		INDT_1 << "/* QGemm */" << std::endl;
		INDT_1 << "const uint32_t LAYER_ID = " << node_id << ";" << std::endl;
		const bool freivalds_enabled = options.freivalds_gemm;

		// Cast C to a 2D view for simple broadcast addressing (like Gemm).
		int C0 = 1, C1 = 1;
		std::string C_idx;
		if (C->is_scalar()) {
			C0 = C1 = 1;
			C_idx = "[0][0]";
		}
		else if (C->rank() == 1) {
			int dim = C->data_dim[0];
			if (dim == M) {
				C0 = M;
				C1 = 1;
			}
			else if (dim == N) {
				C0 = 1;
				C1 = N;
			}
			else if (dim == 1) {
				C0 = 1;
				C1 = 1;
			}
			else {
				ERROR("C dimension mismatch in QGemm");
			}
		}
		else if (C->rank() == 2) {
			C0 = C->data_dim[0];
			C1 = C->data_dim[1];
		}
		else {
			ERROR("C has too many dimensions in QGemm");
		}

		C_idx = "";
		C_idx += (C0 <= 1) ? "[0]" : "[r]";
		C_idx += (C1 <= 1) ? "[0]" : "[c]";
		INDT_1 << "const int32_t (*C_)[" << C1 << "] = (const int32_t(*)[" << C1 << "])C;" << std::endl;

		auto [lower, upper] = Y->get_type_bounds();
		std::string float_dtype = get_input_tensor(1)->data_type_str();

		INDT_1 << "for( uint32_t r=0; r<" << M << "; r++ ) {" << std::endl;

		if (freivalds_enabled) {
			INDT_2 << "/* Freivalds: store row accumulators for multiple checks */" << std::endl;
			INDT_2 << "int32_t acc_row[" << N << "];" << std::endl;
		}

		INDT_2 << "for( uint32_t c=0; c<" << N << "; c++ ) {" << std::endl;

		INDT_3 << "int32_t acc32 = C_" << C_idx << ";" << std::endl;
		INDT_3 << "for( uint32_t i=0; i<" << K << "; i++ ) {" << std::endl;
		std::string B_el = transB ? "B[c][i]" : "B[i][c]";
		INDT_4 << "acc32 += ((int32_t)A[r][i] - (int32_t)a_zero_point[0]) * ((int32_t)" << B_el << " - (int32_t)b_zero_point[0]);" << std::endl;
		INDT_3 << "}" << std::endl;

		// Fault injection on the accumulator before requantization.
		INDT_3 << "/* fault injection */" << std::endl;
		INDT_3 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		INDT_4 << "uint32_t out_idx = (r * " << N << "u) + c;" << std::endl;
		// Interpret FAULT_VALUE as a delta in real (fp32) domain and convert it to an
		// equivalent delta on the int32 accumulator:  delta_acc32 ~= FAULT_VALUE / (a_scale*b_scale).
		INDT_4 << "double eff_scale = (double)a_scale[0] * (double)b_scale[0];" << std::endl;
		INDT_4 << "int32_t fault_delta = 0;" << std::endl;
		INDT_4 << "if( eff_scale != 0.0 ) fault_delta = (int32_t) llround((double)FAULT_VALUE / eff_scale);" << std::endl;
		INDT_4 << "if( FAULT_MODEL==0 ) {" << std::endl;
		INDT_4 << "if( out_idx == FAULT_INDEX ) { acc32 += fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_4 << "/* trivial 2x2 cancelling pattern (top-left at FAULT_INDEX): +e -e; -e +e */" << std::endl;
		INDT_4 << "const uint32_t P = " << N << "u;" << std::endl;
		INDT_4 << "uint32_t base_r = (FAULT_INDEX / P);" << std::endl;
		INDT_4 << "uint32_t base_c = (FAULT_INDEX % P);" << std::endl;
		INDT_4 << "bool ok = (base_r + 1u < (uint32_t)" << M << ") && (base_c + 1u < P);" << std::endl;
		INDT_4 << "int32_t delta = 0;" << std::endl;
		INDT_4 << "if( ok ) {" << std::endl;
		INDT_4 << "if( out_idx == FAULT_INDEX ) delta = +fault_delta;" << std::endl;
		INDT_4 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -fault_delta;" << std::endl;
		INDT_4 << "else if( out_idx == FAULT_INDEX + P ) delta = -fault_delta;" << std::endl;
		INDT_4 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +fault_delta;" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "if( delta != 0 ) { acc32 += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_4 << "/* checkered: repeat the trivial 2x2 cancelling pattern FAULT_N times. */" << std::endl;
		INDT_4 << "const uint32_t P = " << N << "u;" << std::endl;
		INDT_4 << "int32_t delta = 0;" << std::endl;
		INDT_4 << "for( uint32_t rr=0; rr<FAULT_N; rr++ ) {" << std::endl;
		INDT_4 << "uint32_t base = FAULT_INDEX + rr * FAULT_STRIDE;" << std::endl;
		INDT_4 << "uint32_t base_r = (base / P);" << std::endl;
		INDT_4 << "uint32_t base_c = (base % P);" << std::endl;
		INDT_4 << "bool ok = (base_r + 1u < (uint32_t)" << M << ") && (base_c + 1u < P);" << std::endl;
		INDT_4 << "if( !ok ) continue;" << std::endl;
		INDT_4 << "if( out_idx == base ) delta += +fault_delta;" << std::endl;
		INDT_4 << "else if( out_idx == base + 1u ) delta += -fault_delta;" << std::endl;
		INDT_4 << "else if( out_idx == base + P ) delta += -fault_delta;" << std::endl;
		INDT_4 << "else if( out_idx == base + P + 1u ) delta += +fault_delta;" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "if( delta != 0 ) { acc32 += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "}" << std::endl;

		INDT_3 << float_dtype << " scale = (" << float_dtype << ") (a_scale[0] * b_scale[0]) / y_scale[0];" << std::endl;
		INDT_3 << "double scaled = ((double) acc32) * (double) scale;" << std::endl;
		INDT_3 << "scaled = scaled + (double) y_zero_point[0];" << std::endl;
		INDT_3 << "int t = (int) llround(scaled);" << std::endl;
		INDT_3 << "t = MIN(MAX(t, " << lower << "), " << upper << ");" << std::endl;
		INDT_3 << "Y[r][c] = (" << Y->data_type_str() << ") t;" << std::endl;

		if (freivalds_enabled) {
			INDT_3 << "acc_row[c] = acc32;" << std::endl;
		}

		INDT_2 << "}" << std::endl;
		if (freivalds_enabled) {
			INDT_2 << "/* Freivalds verify: (A * (B*r)) + (C*r) == (Y_acc32 * r) */" << std::endl;
			INDT_2 << "for( uint32_t chk=0; chk<" << (options.freivalds_checks ? options.freivalds_checks : 1) << "u; chk++ ) {" << std::endl;
			INDT_3 << "uint32_t freivalds_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ r ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
			INDT_3 << "uint32_t r_vec[" << N << "];" << std::endl;
			INDT_3 << "uint32_t r_any = 0;" << std::endl;
			INDT_3 << "for( uint32_t cc=0; cc<" << N << "; cc++ ) { r_vec[cc] = ABYZFT_randbit(&freivalds_state); r_any |= r_vec[cc]; }" << std::endl;
			INDT_3 << "if( !r_any && " << N << "u>0u ) r_vec[0] = 1u;" << std::endl;
			INDT_3 << "int32_t b_rs[" << K << "];" << std::endl;
			INDT_3 << "for( uint32_t i=0; i<" << K << "; i++ ) b_rs[i] = 0;" << std::endl;
			INDT_3 << "int64_t bias_sum = 0;" << std::endl;
			INDT_3 << "int64_t sumC = 0;" << std::endl;
			INDT_3 << "for( uint32_t cc=0; cc<" << N << "; cc++ ) if( r_vec[cc] ) {" << std::endl;
			INDT_4 << "bias_sum += (int64_t)C_"
			       << ((C0 <= 1) ? "[0]" : "[r]")
			       << ((C1 <= 1) ? "[0]" : "[cc]")
			       << ";" << std::endl;
			INDT_4 << "sumC += (int64_t)acc_row[cc];" << std::endl;
			INDT_4 << "for( uint32_t i=0; i<" << K << "; i++ ) {" << std::endl;
			if (transB) {
				INDT_5 << "b_rs[i] += ((int32_t)B[cc][i] - (int32_t)b_zero_point[0]);" << std::endl;
			}
			else {
				INDT_5 << "b_rs[i] += ((int32_t)B[i][cc] - (int32_t)b_zero_point[0]);" << std::endl;
			}
			INDT_4 << "}" << std::endl;
			INDT_3 << "}" << std::endl;
			INDT_3 << "int64_t pred = bias_sum;" << std::endl;
			INDT_3 << "for( uint32_t i=0; i<" << K << "; i++ ) pred += (int64_t)((int32_t)A[r][i] - (int32_t)a_zero_point[0]) * (int64_t)b_rs[i];" << std::endl;
			INDT_3 << "if( sumC != pred ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
			INDT_2 << "}" << std::endl;
		}

		INDT_1 << "}" << std::endl;
	}
};

} // namespace toC
