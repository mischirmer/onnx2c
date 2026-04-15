/* This file is part of onnx2c.
 *
 * "GEneral Matrix Multiplication"
 * Calulates:
 * Y = alpha*A*B + beta*C
 * optionally trasposing A and/or B first.
 * C need not be of size A*B, but must be
 * 'unidirectionally broadcastable' to A*B.
 */
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
		dst << "\t" << "const uint32_t LAYER_ID = " << node_id << ";" << std::endl;
		dst << "\t" << "float alpha = " << alpha << ";" << std::endl;
		dst << "\t" << "float beta = " << beta << ";" << std::endl;

		std::string A_el = transA ? "A[i][r]" : "A[r][i]";
		std::string B_idx = transB ? "[c][i]" : "[i][c]";
		const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || randomized_enabled;
		const bool freivalds_enabled = options.freivalds_gemm;
		const bool gvfa_enabled = options.gvfa_gemm;
		const uint32_t randomized_checks = freivalds_enabled
		    ? (options.freivalds_checks ? options.freivalds_checks : 1)
		    : (options.gvfa_checks ? options.gvfa_checks : 1);

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

		// Now genereate the calculation source code

		// Loop output rows, columns
		INDT_1 << "for( uint32_t r=0; r<M; r++ )" << std::endl;
		INDT_2 << "{" << std::endl;
		if (checksum_enabled) {
			INDT_2 << "float acc_row[N];" << std::endl;
		}
		INDT_2 << "for( uint32_t c=0; c<N; c++ ) {" << std::endl;

		/* Calculate the matrix muliplication dot inner dot product */
		INDT_3 << type << " ABrc = 0;" << std::endl;
		INDT_3 << "for( uint32_t i=0; i<K; i++ ) {" << std::endl;
		INDT_4 << B->data_type_str() << " B_el = " << constant_acces_code("B" + B_idx) << ";" << std::endl;
		INDT_4 << "ABrc += " << A_el << " * B_el;" << std::endl;
		INDT_3 << "}" << std::endl;

		/* Add scale & bias, store result in output */
		INDT_3 << type << " tmp = ABrc * alpha;" << std::endl;

		if (C) {
			INDT_3 << "tmp += C_" << C_idx << " * beta;" << std::endl;
		}

		INDT_3 << "/* fault injection */" << std::endl;
		INDT_3 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		INDT_4 << "const uint32_t P = (uint32_t)N;" << std::endl;
		INDT_4 << "uint32_t out_idx = ((uint32_t)r * P) + (uint32_t)c;" << std::endl;
		INDT_4 << "if( FAULT_MODEL==0 ) {" << std::endl;
		INDT_4 << "if( out_idx == FAULT_INDEX ) { tmp += FAULT_VALUE; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "} else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_4 << "uint32_t base_r = (FAULT_INDEX / P);" << std::endl;
		INDT_4 << "uint32_t base_c = (FAULT_INDEX % P);" << std::endl;
		INDT_4 << "bool ok = (base_r + 1u < (uint32_t)M) && (base_c + 1u < P);" << std::endl;
		INDT_4 << "float delta = 0.0f;" << std::endl;
		INDT_4 << "if( ok ) {" << std::endl;
		INDT_4 << "if( out_idx == FAULT_INDEX ) delta = +FAULT_VALUE;" << std::endl;
		INDT_4 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -FAULT_VALUE;" << std::endl;
		INDT_4 << "else if( out_idx == FAULT_INDEX + P ) delta = -FAULT_VALUE;" << std::endl;
		INDT_4 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +FAULT_VALUE;" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "if( delta != 0.0f ) { tmp += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "} else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_4 << "float delta = 0.0f;" << std::endl;
		INDT_4 << "for( uint32_t rr=0; rr<FAULT_N; rr++ ) {" << std::endl;
		INDT_4 << "uint32_t base = FAULT_INDEX + rr * FAULT_STRIDE;" << std::endl;
		INDT_4 << "uint32_t base_r = (base / P);" << std::endl;
		INDT_4 << "uint32_t base_c = (base % P);" << std::endl;
		INDT_4 << "bool ok = (base_r + 1u < (uint32_t)M) && (base_c + 1u < P);" << std::endl;
		INDT_4 << "if( !ok ) continue;" << std::endl;
		INDT_4 << "if( out_idx == base ) delta += +FAULT_VALUE;" << std::endl;
		INDT_4 << "else if( out_idx == base + 1u ) delta += -FAULT_VALUE;" << std::endl;
		INDT_4 << "else if( out_idx == base + P ) delta += -FAULT_VALUE;" << std::endl;
		INDT_4 << "else if( out_idx == base + P + 1u ) delta += +FAULT_VALUE;" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "if( delta != 0.0f ) { tmp += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_3 << "}" << std::endl;

		INDT_3 << "Y[r][c] = tmp;" << std::endl;
		if (checksum_enabled) {
			INDT_3 << "acc_row[c] = tmp;" << std::endl;
		}

		INDT_2 << "}" << std::endl;
		if (checksum_enabled) {
			if (freivalds_enabled || gvfa_enabled) {
				INDT_2 << "/* Randomized verify (Freivalds/GVFA): (A * (B*r)) + (C*r) == (Y * r) */" << std::endl;
				INDT_2 << "for( uint32_t chk=0; chk<" << randomized_checks << "u; chk++ ) {" << std::endl;
				INDT_3 << "uint32_t freivalds_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ r ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
				if (freivalds_enabled)
					INDT_3 << "uint8_t r_mask[N];" << std::endl;
				else
					INDT_3 << "float r_vec[N];" << std::endl;
				if (freivalds_enabled) {
					INDT_3 << "uint32_t r_any = 0;" << std::endl;
					INDT_3 << "for( uint32_t cc=0; cc<N; cc++ ) { uint32_t bit = ABYZFT_randbit(&freivalds_state); r_mask[cc] = (uint8_t)bit; r_any |= bit; }" << std::endl;
					INDT_3 << "if( !r_any && N>0u ) r_mask[0] = 1u;" << std::endl;
				}
				else {
					INDT_3 << "for( uint32_t cc=0; cc<N; cc++ ) r_vec[cc] = ABYZFT_randn(&freivalds_state);" << std::endl;
				}
				INDT_3 << "float b_rs[K];" << std::endl;
				INDT_3 << "for( uint32_t i=0; i<K; i++ ) b_rs[i] = 0.0f;" << std::endl;
				INDT_3 << "float bias_sum = 0.0f;" << std::endl;
				INDT_3 << "float sumC = 0.0f;" << std::endl;
				INDT_3 << "for( uint32_t cc=0; cc<N; cc++ ) {" << std::endl;
				if (freivalds_enabled) {
					INDT_4 << "if( !r_mask[cc] ) continue;" << std::endl;
					if (C) {
						INDT_4 << "bias_sum += C_"
						       << ((C0 <= 1) ? "[0]" : "[r]")
						       << ((C1 <= 1) ? "[0]" : "[cc]")
						       << " * beta;" << std::endl;
					}
					INDT_4 << "sumC += acc_row[cc];" << std::endl;
					INDT_4 << "for( uint32_t i=0; i<K; i++ ) {" << std::endl;
					if (transB) {
						INDT_5 << "b_rs[i] += " << constant_acces_code("B[cc][i]") << ";" << std::endl;
					}
					else {
						INDT_5 << "b_rs[i] += " << constant_acces_code("B[i][cc]") << ";" << std::endl;
					}
					INDT_4 << "}" << std::endl;
				}
				else {
					if (C) {
						INDT_4 << "bias_sum += C_"
						       << ((C0 <= 1) ? "[0]" : "[r]")
						       << ((C1 <= 1) ? "[0]" : "[cc]")
						       << " * beta * r_vec[cc];" << std::endl;
					}
					INDT_4 << "sumC += acc_row[cc] * r_vec[cc];" << std::endl;
					INDT_4 << "for( uint32_t i=0; i<K; i++ ) {" << std::endl;
					if (transB) {
						INDT_5 << "b_rs[i] += " << constant_acces_code("B[cc][i]") << " * r_vec[cc];" << std::endl;
					}
					else {
						INDT_5 << "b_rs[i] += " << constant_acces_code("B[i][cc]") << " * r_vec[cc];" << std::endl;
					}
					INDT_4 << "}" << std::endl;
				}
				INDT_3 << "}" << std::endl;
				INDT_3 << "float pred = bias_sum;" << std::endl;
				INDT_3 << "for( uint32_t i=0; i<K; i++ ) pred += alpha * (" << A_el << ") * b_rs[i];" << std::endl;
				INDT_3 << "float diff = fabsf(sumC - pred);" << std::endl;
				INDT_3 << "float tol = " << options.abft_eps << "f * (fabsf(pred) + 1.0f);" << std::endl;
				INDT_3 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
				INDT_2 << "}" << std::endl;
			}
			else {
				INDT_2 << "/* ABFT verify: sum(Y_row) == alpha*(A_row*(B*1)) + beta*sum(C_row) */" << std::endl;
				INDT_2 << "float b_s[K];" << std::endl;
				INDT_2 << "for( uint32_t i=0; i<K; i++ ) b_s[i] = 0.0f;" << std::endl;
				INDT_2 << "float bias_sum = 0.0f;" << std::endl;
				INDT_2 << "float sumC = 0.0f;" << std::endl;
				INDT_2 << "for( uint32_t cc=0; cc<N; cc++ ) {" << std::endl;
				if (C) {
					INDT_3 << "bias_sum += C_"
					       << ((C0 <= 1) ? "[0]" : "[r]")
					       << ((C1 <= 1) ? "[0]" : "[cc]")
					       << " * beta;" << std::endl;
				}
				INDT_3 << "sumC += acc_row[cc];" << std::endl;
				INDT_3 << "for( uint32_t i=0; i<K; i++ ) {" << std::endl;
				if (transB) {
					INDT_4 << "b_s[i] += " << constant_acces_code("B[cc][i]") << ";" << std::endl;
				}
				else {
					INDT_4 << "b_s[i] += " << constant_acces_code("B[i][cc]") << ";" << std::endl;
				}
				INDT_3 << "}" << std::endl;
				INDT_2 << "}" << std::endl;
				INDT_2 << "float pred = bias_sum;" << std::endl;
				INDT_2 << "for( uint32_t i=0; i<K; i++ ) pred += alpha * (" << A_el << ") * b_s[i];" << std::endl;
				INDT_2 << "float diff = fabsf(sumC - pred);" << std::endl;
				INDT_2 << "float tol = " << (options.abyzft_gemm ? (options.abft_eps * 0.1f) : options.abft_eps) << "f * (fabsf(pred) + 1.0f);" << std::endl;
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
