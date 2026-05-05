/* This file is part of onnx2c.
 *
 * MatMul node implementation.
 */

#include "matmul.h"
#include "../options.h"

namespace toC {

void MatMul::resolve(void)
{
	Tensor* a = get_input_tensor(0);
	Tensor* b = get_input_tensor(1);

	if (a->data_type != b->data_type) {
		ERROR("Data types of A and B must match in MatMul");
	}

	name_input(0, "A");
	name_input(1, "B");

	Tensor* y = new Tensor;
	y->data_dim = resolve_shape();
	y->data_type = a->data_type;
	register_output(y, "Y");
}

void MatMul::print(std::ostream& dst) const
{
	const Tensor* A = get_input_tensor(0);
	const Tensor* B = get_input_tensor(1);
	const Tensor* Y = get_output_tensor(0);
	// AByzFT uses ABFT-style (sum-based) output checksums — its input scaling already provides
	// the randomization for Byzantine fault detection, and ABFT-style allows compile-time
	// weight checksum precomputation.
	const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
	const bool checksum_enabled = options.abft_gemm || randomized_enabled || options.abyzft_gemm;
	const bool freivalds_enabled = options.freivalds_gemm;
	const uint32_t randomized_checks = freivalds_enabled
	    ? (options.freivalds_checks ? options.freivalds_checks : 1)
	    : (options.gvfa_checks ? options.gvfa_checks : 1);

	if (A->rank() != 2 || B->rank() != 2 || Y->rank() != 2) {
		AbstractMatMul::print(dst);
		return;
	}

	const int M = A->data_dim[0];
	const int K = A->data_dim[1];
	const int N = B->data_dim[1];

	INDT_1 << "/* MatMul */" << std::endl;
	INDT_1 << "const uint32_t LAYER_ID = " << sweep_layer_id << ";" << std::endl;

	if (checksum_enabled && !randomized_enabled) {
		const bool ct = options.abft_weight_checksums_compiletime && B->isConst && B->data_buffer;
		if (ct) {
			INDT_1 << "/* Compile-time ABFT checksums */" << std::endl;
			INDT_1 << "static const float b_rs_cache[" << K << "] = {" << std::endl;
			float* bd = (float*)B->data_buffer;
			INDT_2 << "";
			for (int k = 0; k < K; k++) {
				double s = 0.0;
				for (int j = 0; j < N; j++) s += (double)bd[k * N + j];
				if (k) dst << ", ";
				dst << (float)s;
			}
			dst << "};" << std::endl;
		} else {
			INDT_1 << "float b_rs_cache[" << K << "];" << std::endl;
			INDT_1 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) {" << std::endl;
			INDT_2 << "b_rs_cache[kk2] = 0.0f;" << std::endl;
			INDT_2 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) b_rs_cache[kk2] += B[kk2][cc];" << std::endl;
			INDT_1 << "}" << std::endl;
		}
	}

	// For Freivalds/GVFA: pre-compute random projections and weight projections ONCE before the
	// row loop. The original code recomputed b_rs[K] per row with a row-dependent seed, causing
	// O(M*N*K) overhead matching the matmul itself. With a row-independent seed, b_rs is computed
	// once O(N*K), and verification inside the row loop is just O(K) per row.
	if (checksum_enabled && randomized_enabled) {
		if (freivalds_enabled) {
			INDT_1 << "uint8_t r_cache[" << randomized_checks << "][" << N << "];" << std::endl;
		} else {
			INDT_1 << "float r_cache[" << randomized_checks << "][" << N << "];" << std::endl;
		}
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
			INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) b_rs_cache[chk][kk2] += (double)B[kk2][cc];" << std::endl;
		} else {
			INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) b_rs_cache[chk][kk2] += (double)B[kk2][cc] * (double)r_cache[chk][cc];" << std::endl;
		}
		INDT_2 << "}" << std::endl;
		INDT_1 << "}" << std::endl;
	}

	if (options.abyzft_gemm) {
		INDT_1 << "float abyzft_scaleA[" << M << "];" << std::endl;
		INDT_1 << "{ uint32_t _s = 0xDEADBEEFu ^ LAYER_ID; for(uint32_t _r=0; _r<" << M << "u; _r++) abyzft_scaleA[_r] = 0.25f + 3.75f * ABYZFT_rand01(&_s); }" << std::endl;
		INDT_1 << "float abyzft_scaleB[" << N << "];" << std::endl;
		INDT_1 << "{ uint32_t _s = 0xCAFEBABEu ^ LAYER_ID; for(uint32_t _c=0; _c<" << N << "u; _c++) abyzft_scaleB[_c] = 0.25f + 3.75f * ABYZFT_rand01(&_s); }" << std::endl;
	}

	INDT_1 << "for( uint32_t r=0; r<" << M << "u; r++ ) {" << std::endl;
	if (checksum_enabled && randomized_enabled) {
		INDT_2 << "float acc_row[" << N << "];" << std::endl;
	}
	if (checksum_enabled && !randomized_enabled) {
		INDT_2 << "float sumC = 0.0f;" << std::endl;
	}

	INDT_2 << "for( uint32_t c=0; c<" << N << "u; c++ ) {" << std::endl;

	INDT_3 << "float acc = 0.0f;" << std::endl;
	INDT_3 << "for( uint32_t k=0; k<" << K << "u; k++ ) {" << std::endl;
	INDT_4 << "acc += A[r][k] * B[k][c];" << std::endl;
	INDT_3 << "}" << std::endl;
	if (options.abyzft_gemm) {
		INDT_3 << "float _sAB = abyzft_scaleA[r] * abyzft_scaleB[c];" << std::endl;
		INDT_3 << "float acc_scaled = acc * _sAB;" << std::endl;
	}

	INDT_3 << "/* fault injection */" << std::endl;
	INDT_3 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
	INDT_4 << "const uint32_t P = " << N << "u;" << std::endl;
	INDT_4 << "uint32_t out_idx = r * P + c;" << std::endl;
	INDT_4 << "if( FAULT_MODEL==0 ) {" << std::endl;
	if (options.abyzft_gemm)
		INDT_4 << "if( out_idx == FAULT_INDEX ) { acc_scaled += FAULT_VALUE; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
	else
		INDT_4 << "if( out_idx == FAULT_INDEX ) { acc += FAULT_VALUE; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
	INDT_4 << "}" << std::endl;
	INDT_4 << "else if( FAULT_MODEL==1 ) {" << std::endl;
	INDT_4 << "uint32_t base_r = (FAULT_INDEX / P);" << std::endl;
	INDT_4 << "uint32_t base_c = (FAULT_INDEX % P);" << std::endl;
	INDT_4 << "float delta = 0.0f;" << std::endl;
	INDT_4 << "bool ok = (base_r + 1u < " << M << "u) && (base_c + 1u < P);" << std::endl;
	INDT_4 << "if( ok ) {" << std::endl;
	INDT_5 << "if( out_idx == FAULT_INDEX ) delta = +FAULT_VALUE;" << std::endl;
	INDT_5 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -FAULT_VALUE;" << std::endl;
	INDT_5 << "else if( out_idx == FAULT_INDEX + P ) delta = -FAULT_VALUE;" << std::endl;
	INDT_5 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +FAULT_VALUE;" << std::endl;
	INDT_4 << "}" << std::endl;
	if (options.abyzft_gemm)
		INDT_4 << "if( delta != 0.0f ) { acc_scaled += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
	else
		INDT_4 << "if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
	INDT_4 << "}" << std::endl;
	INDT_4 << "else if( FAULT_MODEL==2 ) {" << std::endl;
	INDT_4 << "float delta = 0.0f;" << std::endl;
	INDT_4 << "for( uint32_t rr=0; rr<FAULT_N; rr++ ) {" << std::endl;
	INDT_5 << "uint32_t base = FAULT_INDEX + rr * FAULT_STRIDE;" << std::endl;
	INDT_5 << "uint32_t base_r = (base / P);" << std::endl;
	INDT_5 << "uint32_t base_c = (base % P);" << std::endl;
	INDT_5 << "bool ok = (base_r + 1u < " << M << "u) && (base_c + 1u < P);" << std::endl;
	INDT_5 << "if( !ok ) continue;" << std::endl;
	INDT_5 << "if( out_idx == base ) delta += +FAULT_VALUE;" << std::endl;
	INDT_5 << "else if( out_idx == base + 1u ) delta += -FAULT_VALUE;" << std::endl;
	INDT_5 << "else if( out_idx == base + P ) delta += -FAULT_VALUE;" << std::endl;
	INDT_5 << "else if( out_idx == base + P + 1u ) delta += +FAULT_VALUE;" << std::endl;
	INDT_4 << "}" << std::endl;
	if (options.abyzft_gemm)
		INDT_4 << "if( delta != 0.0f ) { acc_scaled += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
	else
		INDT_4 << "if( delta != 0.0f ) { acc += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
	INDT_4 << "}" << std::endl;
	INDT_3 << "}" << std::endl;

	if (options.abyzft_gemm) {
		INDT_3 << "acc = (_sAB != 0.0f) ? acc_scaled / _sAB : acc;" << std::endl;
	}

	if (checksum_enabled && randomized_enabled) {
		INDT_3 << "acc_row[c] = acc;" << std::endl;
	} else if (checksum_enabled) {
		INDT_3 << "sumC += acc;" << std::endl;
	}

	INDT_3 << "Y[r][c] = acc;" << std::endl;
	INDT_2 << "}" << std::endl;

	if (checksum_enabled) {
		if (randomized_enabled) {
			// Use pre-computed r_cache and b_rs_cache (generated before the row loop).
			// sumC_rand is O(N) per row; pred dot-product is O(K) per row — no O(N*K) recomputation.
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
			INDT_3 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) pred += (double)A[r][kk2] * b_rs_cache[chk][kk2];" << std::endl;
			INDT_3 << "double diff = fabs(pred - sumC_rand);" << std::endl;
			INDT_3 << "double tol = " << (options.abyzft_gemm ? (options.abft_eps * 0.1f) : options.abft_eps) << " * (fabs(pred) + 1.0);" << std::endl;
			INDT_3 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
			INDT_2 << "}" << std::endl;
		} else {
			const bool ct = options.abft_weight_checksums_compiletime && B->isConst && B->data_buffer;
			if (ct) {
				INDT_2 << "const float* b_rs = b_rs_cache;" << std::endl;
			}
			INDT_2 << "float pred = 0.0f;" << std::endl;
			INDT_2 << "for( uint32_t kk2=0; kk2<" << K << "u; kk2++ ) pred += A[r][kk2] * b_rs_cache[kk2];" << std::endl;
			INDT_2 << "float diff = fabsf(sumC - pred);" << std::endl;
			INDT_2 << "float tol = " << (options.abyzft_gemm ? (options.abft_eps * 0.1f) : options.abft_eps) << "f * (fabsf(pred) + 1.0f);" << std::endl;
			INDT_2 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
		}
	}

	INDT_1 << "}" << std::endl;
}

void MatMul::print_multiply_accumulate(std::ostream& dst,
                                       const std::string& y_idx,
                                       const std::string& a_idx,
                                       const std::string& b_idx) const
{
	INDT_4 << y_idx << " += " << a_idx << " * " << b_idx << ";" << std::endl;
}

} // namespace toC
