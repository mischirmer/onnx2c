/* This file is part of onnx2c.
 *
 * "GEneral Matrix Multiplication"
 * Calulates:
 * Y = alpha*A*B + beta*C
 * optionally trasposing A and/or B first.
 * C need not be of size A*B, but must be
 * 'unidirectionally broadcastable' to A*B.
 */

#include "../options.h"
#include <cmath>

namespace toC {

class Gemm : public Node {
	public:
	Gemm()
	{
		op_name = "Gemm";
		alpha = beta = 1;
		transA = transB = 0;
	}

	/* Node attributes */
	float alpha;
	float beta;
	int transA; // boolean for 'do the tranpose'
	int transB;

	/* Parse attributes, if this node has them. */
	virtual void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

			if (a.name() == "alpha")
				alpha = parse_attribute_float(a);
			else if (a.name() == "beta")
				beta = parse_attribute_float(a);
			else if (a.name() == "transA")
				transA = parse_attribute_int(a);
			else if (a.name() == "transB")
				transB = parse_attribute_int(a);
			else
				ERROR("unknown attribute: " << a.name());
		}
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream& dst) const override
	{
		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(1);
		const Tensor* C = get_number_of_inputs() > 2 ? get_input_tensor(2) : nullptr;
		//	int A1 = A->data_dim[1];
		int C0, C1;
		C0 = C1 = 0;
		if (C && C->is_scalar() == false) {
			C0 = C->data_dim[0];
			if (C->rank() > 1) {
				C1 = C->data_dim[1];
			}
		}

		int M = transA ? A->data_dim[1] : A->data_dim[0]; // row
		int K = transA ? A->data_dim[0] : A->data_dim[1]; // inner
		int N = transB ? B->data_dim[0] : B->data_dim[1]; // column
		std::string type = A->data_type_str();
		const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || randomized_enabled;
		const bool freivalds_enabled = options.freivalds_gemm;
		const uint32_t randomized_checks = freivalds_enabled
		    ? (options.freivalds_checks ? options.freivalds_checks : 1)
		    : (options.gvfa_checks ? options.gvfa_checks : 1);
		INDT_1 << "const uint32_t LAYER_ID = " << sweep_layer_id << ";" << std::endl;

		// Documentation if someone is reading the code
		dst << "\t/* Gemm */" << std::endl;
		dst << "\t/* alpha   = " << alpha << std::endl;
		dst << "\t   beta    = " << beta << std::endl;
		dst << "\t   transA  = " << transA << std::endl;
		dst << "\t   transB  = " << transB << std::endl;
		dst << "\t */" << std::endl;

		// Helper variables to make the code (both this and generated) cleaner
		dst << "\t" << "const int M = " << M << ";" << std::endl;
		dst << "\t" << "const int K = " << K << ";" << std::endl;
		dst << "\t" << "const int N = " << N << ";" << std::endl;
		dst << "\t" << "float alpha = " << alpha << ";" << std::endl;
		dst << "\t" << "float beta = " << beta << ";" << std::endl;

		std::string A_el = transA ? "A[i][r]" : "A[r][i]";
		std::string B_idx = transB ? "[c][i]" : "[i][c]";

		// Cast optional C matrix to generated variable
		// "C_[M][N]"
		std::string C_idx;
		if (C) {
			C_idx = "";
			int dim;
			switch (C->rank()) {
				case 0:
					C0 = C1 = 0;
					break;
				case 1:
					dim = C->data_dim[0];
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
						ERROR("C dimension mismatch in Gemm");
					}
					break;
				case 2:
					C0 = C->data_dim[0];
					C1 = C->data_dim[1];
					break;
				default:
					ERROR("C has too many dimensions in Gemm");
			}
			if (C0 <= 1)
				C_idx += "[0]";
			else
				C_idx += "[r]";
			if (C1 <= 1)
				C_idx += "[0]";
			else
				C_idx += "[c]";
			INDT_1 << type << " (*C_)[" << C1 << "]  = (" << type << "(*)[" << C1 << "])C;" << std::endl;
		}

		if (checksum_enabled && !randomized_enabled) {
			const bool ct = options.abft_weight_checksums_compiletime && B->isConst && B->data_buffer;
			if (ct) {
				INDT_1 << "/* Compile-time ABFT checksums */" << std::endl;
				INDT_1 << "static const float b_rs_cache[" << K << "] = {" << std::endl;
				float* bd = (float*)B->data_buffer;
				INDT_2 << "";
				for (int k = 0; k < K; k++) {
					double s = 0.0;
					for (int j = 0; j < N; j++) s += (double)(transB ? bd[j * K + k] : bd[k * N + j]);
					if (k) dst << ", ";
					dst << (float)s;
				}
				dst << "};" << std::endl;
			} else {
				INDT_1 << "float b_rs_cache[" << K << "];" << std::endl;
				INDT_1 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) {" << std::endl;
				INDT_2 << "b_rs_cache[kk2] = 0.0f;" << std::endl;
				if (transB)
					INDT_2 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) b_rs_cache[kk2] += B[cc][kk2];" << std::endl;
				else
					INDT_2 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) b_rs_cache[kk2] += B[kk2][cc];" << std::endl;
				INDT_1 << "}" << std::endl;
			}
		}

		if (checksum_enabled && randomized_enabled) {
			if (freivalds_enabled)
				INDT_1 << "uint8_t r_cache[" << randomized_checks << "][" << N << "];" << std::endl;
			else
				INDT_1 << "float r_cache[" << randomized_checks << "][" << N << "];" << std::endl;
			INDT_1 << "double b_rs_cache[" << randomized_checks << "][" << K << "];" << std::endl;
			INDT_1 << "for( uint32_t chk=0; chk<" << randomized_checks << "u; chk++ ) {" << std::endl;
			INDT_2 << "uint32_t rand_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ (uint32_t)(chk * 0x85EBCA6Bu));" << std::endl;
			if (freivalds_enabled) {
				INDT_2 << "uint32_t r_any = 0;" << std::endl;
				INDT_2 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) { uint32_t bit = ABYZFT_randbit(&rand_state); r_cache[chk][cc] = (uint8_t)bit; r_any |= bit; }" << std::endl;
				INDT_2 << "if( !r_any && " << N << "u>0u ) r_cache[chk][0] = 1u;" << std::endl;
			} else {
				INDT_2 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) r_cache[chk][cc] = ABYZFT_randn(&rand_state);" << std::endl;
			}
			INDT_2 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) b_rs_cache[chk][kk2] = 0.0;" << std::endl;
			INDT_2 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) {" << std::endl;
			if (freivalds_enabled) {
				INDT_3 << "if( !r_cache[chk][cc] ) continue;" << std::endl;
				if (transB)
					INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) b_rs_cache[chk][kk2] += (double)B[cc][kk2];" << std::endl;
				else
					INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) b_rs_cache[chk][kk2] += (double)B[kk2][cc];" << std::endl;
			} else {
				if (transB)
					INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) b_rs_cache[chk][kk2] += (double)B[cc][kk2] * (double)r_cache[chk][cc];" << std::endl;
				else
					INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) b_rs_cache[chk][kk2] += (double)B[kk2][cc] * (double)r_cache[chk][cc];" << std::endl;
			}
			INDT_2 << "}" << std::endl;
			INDT_1 << "}" << std::endl;
		}

		// Now genereate the calculation source code
		INDT_1 << "for( uint32_t r=0; r<M; r++ ) {" << std::endl;
		if (checksum_enabled && randomized_enabled)
			INDT_2 << "float acc_row[" << N << "];" << std::endl;
		if (checksum_enabled && !randomized_enabled)
			INDT_2 << "float sumC = 0.0f;" << std::endl;

		INDT_2 << "for( uint32_t c=0; c<N; c++ ) {" << std::endl;
		INDT_3 << type << " ABrc = 0;" << std::endl;
		INDT_3 << "for( uint32_t i=0; i<K; i++ ) {" << std::endl;
		INDT_4 << B->data_type_str() << " B_el = " << constant_acces_code("B" + B_idx) << ";" << std::endl;
		INDT_4 << "ABrc += " << A_el << " * B_el;" << std::endl;
		INDT_3 << "}" << std::endl;

		if (checksum_enabled && randomized_enabled)
			INDT_3 << "acc_row[c] = ABrc;" << std::endl;
		else if (checksum_enabled)
			INDT_3 << "sumC += ABrc;" << std::endl;

		INDT_3 << type << " tmp = ABrc * alpha;" << std::endl;
		if (C)
			INDT_3 << "tmp += C_" << C_idx << " * beta;" << std::endl;
		INDT_3 << "Y[r][c] = tmp;" << std::endl;
		INDT_2 << "}" << std::endl;

		if (checksum_enabled) {
			if (randomized_enabled) {
				if (freivalds_enabled)
					INDT_2 << "/* Freivalds verify: r^T C_row == A_row * (B r) */" << std::endl;
				else
					INDT_2 << "/* GVFA verify: r^T C_row ~= A_row * (B r) */" << std::endl;
				INDT_2 << "for( uint32_t chk=0; chk<" << randomized_checks << "u; chk++ ) {" << std::endl;
				INDT_3 << "double sumC_rand = 0.0;" << std::endl;
				INDT_3 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) {" << std::endl;
				if (freivalds_enabled) {
					INDT_4 << "if( !r_cache[chk][cc] ) continue;" << std::endl;
					INDT_4 << "sumC_rand += (double)acc_row[cc];" << std::endl;
				} else {
					INDT_4 << "sumC_rand += (double)acc_row[cc] * (double)r_cache[chk][cc];" << std::endl;
				}
				INDT_3 << "}" << std::endl;
				INDT_3 << "double pred = 0.0;" << std::endl;
				if (transA)
					INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) pred += (double)A[kk2][r] * b_rs_cache[chk][kk2];" << std::endl;
				else
					INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) pred += (double)A[r][kk2] * b_rs_cache[chk][kk2];" << std::endl;
				INDT_3 << "double diff = fabs(pred - sumC_rand);" << std::endl;
				INDT_3 << "double tol = " << options.abft_eps << " * (fabs(pred) + 1.0);" << std::endl;
				INDT_3 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
				INDT_2 << "}" << std::endl;
			} else {
				INDT_2 << "/* ABFT verify (float domain): sum(C_row) == A_row * (B 1) */" << std::endl;
				INDT_2 << "float pred = 0.0f;" << std::endl;
				if (transA)
					INDT_2 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) pred += A[kk2][r] * b_rs_cache[kk2];" << std::endl;
				else
					INDT_2 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) pred += A[r][kk2] * b_rs_cache[kk2];" << std::endl;
				INDT_2 << "float diff = fabsf(sumC - pred);" << std::endl;
				INDT_2 << "float tol = " << options.abft_eps << "f * (fabsf(pred) + 1.0f);" << std::endl;
				INDT_2 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
			}
		}

		INDT_1 << "}" << std::endl;
	}

	/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
	virtual void resolve(void) override
	{
		if (get_number_of_inputs() < 2)
			ERROR("Not enough inputs");

		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(1);
		name_input(0, "A");
		name_input(1, "B");

		if (get_number_of_inputs() == 3) {
			name_input(2, "C");
		}

		// output dimensions - see the specification
		int M = transA ? A->data_dim[1] : A->data_dim[0];
		int N = transB ? B->data_dim[0] : B->data_dim[1];

		/* Create output tensors.
		 * Set data dimensions and data type for the created tensors. */
		Tensor* t = new Tensor;
		t->data_dim.push_back(M);
		t->data_dim.push_back(N);
		t->data_type = A->data_type;
		register_output(t, "Y");
	}
};
} // namespace toC
