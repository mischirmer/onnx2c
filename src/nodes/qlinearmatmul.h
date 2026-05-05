/* This file is part of onnx2c.
 *
 * QLinearMatMul
 *
 */

#include "abstractmatmul.h"
#include "../options.h"
#include <cmath>

namespace toC {
class QLinearMatMul : public AbstractMatMul {
	public:
	QLinearMatMul()
	{
		op_name = "QLinearMatMul";
	}

	Tensor* get_a() const override { return get_input_tensor(0); }
	Tensor* get_b() const override { return get_input_tensor(3); }

	void print_multiply_accumulate(std::ostream& dst,
	                               const std::string&,
	                               const std::string&,
	                               const std::string&) const override
	{
		(void)dst;
	}

	void name_scalar_input(unsigned input_no, std::string name)
	{
		name_input(input_no, name);
		if (!(get_input_tensor(input_no)->data_dim.size() == 0 ||
		      (get_input_tensor(input_no)->data_dim.size() == 1 && get_input_tensor(input_no)->data_dim[0] == 1))) {
			ERROR(name << " must be scalar");
		}
	}

	void print(std::ostream& dst) const override
	{
		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(3);
		const Tensor* Y = get_output_tensor(0);

		if (A->rank() != 2 || B->rank() != 2 || Y->rank() != 2) {
			ERROR("Unimplemented: QLinearMatMul only supports rank-2 tensors");
		}

		const int M = A->data_dim[0];
		const int K = A->data_dim[1];
		const int N = B->data_dim[1];
		const bool randomized_enabled = options.freivalds_gemm || options.gvfa_gemm;
		const bool checksum_enabled = options.abft_gemm || options.abyzft_gemm || randomized_enabled;
		const bool freivalds_enabled = options.freivalds_gemm;
		const bool gvfa_enabled = options.gvfa_gemm;
		const uint32_t randomized_checks = freivalds_enabled
		    ? (options.freivalds_checks ? options.freivalds_checks : 1)
		    : (options.gvfa_checks ? options.gvfa_checks : 1);
		auto [lower, upper] = Y->get_type_bounds();
		std::string float_dtype = get_input_tensor(1)->data_type_str();
		const bool unsigned_quant = typeConstraint_unsigned_integers(A);
		const char* scale_picker = unsigned_quant ? "ABYZFT_pick_scale_u8" : "ABYZFT_pick_scale_s8";
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
		bool use_compiletime_abyzft_bbase = false;
		int32_t b_zp_const = 0;
		if (options.abyzft_gemm && B->isConst) {
			use_compiletime_abyzft_bbase = supports_compiletime_i32_read(B);
			const Tensor* b_zp_t = get_input_tensor(5);
			if (!b_zp_t->isConst || !supports_compiletime_i32_read(b_zp_t))
				use_compiletime_abyzft_bbase = false;
			else
				b_zp_const = get_tensor_elem_i32(b_zp_t, 0);
		}
		bool use_abyzft_i32_accum = false;
		if (options.abyzft_gemm) {
			const Tensor* a_zp_t = get_input_tensor(2);
			const Tensor* b_zp_t = get_input_tensor(5);
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
			if (a_zp_t->isConst && b_zp_t->isConst &&
			    supports_compiletime_i32_read(a_zp_t) &&
			    supports_compiletime_i32_read(b_zp_t)) {
				int32_t a_zp = get_tensor_elem_i32(a_zp_t, 0);
				int32_t b_zp = get_tensor_elem_i32(b_zp_t, 0);
				auto [a_lo, a_hi] = type_bounds_i64(A);
				auto [b_lo, b_hi] = type_bounds_i64(B);
				const int64_t max_abs_a = std::max(abs64((int64_t)a_lo - a_zp), abs64((int64_t)a_hi - a_zp));
				const int64_t max_abs_b = std::max(abs64((int64_t)b_lo - b_zp), abs64((int64_t)b_hi - b_zp));
				const int64_t max_scale = 8;
				const int64_t worst = (int64_t)K * (max_abs_a * max_scale) * (max_abs_b * max_scale);
				use_abyzft_i32_accum = (worst <= 2147483647ll);
			}
		}

		INDT_1 << "/* QLinearMatMul */" << std::endl;
		INDT_1 << "const uint32_t LAYER_ID = " << sweep_layer_id << ";" << std::endl;
		if (options.abyzft_gemm && use_compiletime_abyzft_bbase) {
			INDT_1 << "static const int16_t b_base_cache[" << K << "][" << N << "] = {" << std::endl;
			for (int i = 0; i < K; i++) {
				INDT_2 << "{";
				for (int c = 0; c < N; c++) {
					if (c > 0) dst << ", ";
					uint64_t idx = static_cast<uint64_t>(i) * static_cast<uint64_t>(N) + static_cast<uint64_t>(c);
					dst << static_cast<int16_t>(get_tensor_elem_i32(B, idx) - b_zp_const);
				}
				dst << "}";
				if (i + 1 < K) dst << ",";
				dst << std::endl;
			}
			INDT_1 << "};" << std::endl;
		}
		INDT_1 << "for( uint32_t r=0; r<" << M << "u; r++ ) {" << std::endl;

		if (checksum_enabled) {
			INDT_2 << "int32_t acc_row[" << N << "];" << std::endl;
		}

		INDT_2 << "for( uint32_t c=0; c<" << N << "u; c++ ) {" << std::endl;
		if (options.abyzft_gemm) {
			INDT_3 << "uint32_t abyzft_state = (uint32_t)(LAYER_ID ^ r ^ c);" << std::endl;
			INDT_3 << "int32_t abyzft_scaleA = (int32_t)" << scale_picker << "(&abyzft_state);" << std::endl;
			INDT_3 << "int32_t abyzft_scaleB = (int32_t)" << scale_picker << "(&abyzft_state);" << std::endl;
			INDT_3 << "int64_t abyzft_scaleAB = (int64_t)abyzft_scaleA * (int64_t)abyzft_scaleB;" << std::endl;
			INDT_3 << "int64_t abyzft_scaleAB_mag = (abyzft_scaleAB < 0) ? -abyzft_scaleAB : abyzft_scaleAB;" << std::endl;
			INDT_3 << "int32_t abyzft_scaleAB_sign = (abyzft_scaleAB < 0) ? -1 : 1;" << std::endl;
			INDT_3 << "uint32_t abyzft_shiftAB = ABYZFT_pow2_shift_u64((uint64_t)abyzft_scaleAB_mag);" << std::endl;
			if (use_abyzft_i32_accum)
				INDT_3 << "int32_t acc_scaled = 0;" << std::endl;
			else
				INDT_3 << "int64_t acc_scaled = 0;" << std::endl;
			INDT_3 << "int16_t lhs_scaled_row[" << K << "];" << std::endl;
			INDT_3 << "int16_t rhs_scaled_col[" << K << "];" << std::endl;
			INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) lhs_scaled_row[i] = (int16_t)(((int32_t)A[r][i] - (int32_t)a_zero_point[0]) * abyzft_scaleA);" << std::endl;
			if (use_compiletime_abyzft_bbase)
				INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) rhs_scaled_col[i] = (int16_t)(b_base_cache[i][c] * abyzft_scaleB);" << std::endl;
			else
				INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) rhs_scaled_col[i] = (int16_t)(((int32_t)B[i][c] - (int32_t)b_zero_point[0]) * abyzft_scaleB);" << std::endl;
			INDT_3 << "int32_t acc32 = 0;" << std::endl;
			INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) {" << std::endl;
			if (use_abyzft_i32_accum)
				INDT_4 << "acc_scaled += (int32_t)lhs_scaled_row[i] * (int32_t)rhs_scaled_col[i];" << std::endl;
			else
				INDT_4 << "acc_scaled += (int64_t)lhs_scaled_row[i] * (int64_t)rhs_scaled_col[i];" << std::endl;
			INDT_3 << "}" << std::endl;
			if (use_abyzft_i32_accum)
				INDT_3 << "int64_t acc_scaled_fault = (int64_t)acc_scaled;" << std::endl;
		}
		else {
			INDT_3 << "int32_t acc32 = 0;" << std::endl;
			INDT_3 << "int32_t acc32_check = 0;" << std::endl;
			INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) {" << std::endl;
			INDT_4 << "acc32 += ((int32_t)A[r][i] - (int32_t)a_zero_point[0]) * ((int32_t)B[i][c] - (int32_t)b_zero_point[0]);" << std::endl;
			INDT_3 << "}" << std::endl;
			INDT_3 << "acc32_check = acc32;" << std::endl;
		}

		INDT_3 << "/* fault injection */" << std::endl;
		INDT_3 << "if( FAULT_ENABLED && (FAULT_LAYER_ID==LAYER_ID || FAULT_LAYER_ID==0xFFFFFFFFu) ) {" << std::endl;
		INDT_4 << "uint32_t out_idx = (r * " << N << "u) + c;" << std::endl;
		INDT_4 << "double eff_scale = (double)a_scale[0] * (double)b_scale[0];" << std::endl;
		INDT_4 << "int32_t fault_delta = 0;" << std::endl;
		INDT_4 << "if( eff_scale != 0.0 ) fault_delta = (int32_t) llround((double)FAULT_VALUE / eff_scale);" << std::endl;
		if (options.abyzft_gemm)
			INDT_4 << "int64_t scaled_fault_delta = (int64_t)fault_delta;" << std::endl;
		INDT_4 << "if( FAULT_MODEL==0 ) {" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( out_idx == FAULT_INDEX ) { " << (use_abyzft_i32_accum ? "acc_scaled_fault" : "acc_scaled") << " += scaled_fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( out_idx == FAULT_INDEX ) { acc32 += fault_delta; acc32_check += fault_delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_4 << "else if( FAULT_MODEL==1 ) {" << std::endl;
		INDT_5 << "const uint32_t P = " << N << "u;" << std::endl;
		INDT_5 << "uint32_t base_r = (FAULT_INDEX / P);" << std::endl;
		INDT_5 << "uint32_t base_c = (FAULT_INDEX % P);" << std::endl;
		INDT_5 << "bool ok = (base_r + 1u < (uint32_t)" << M << ") && (base_c + 1u < P);" << std::endl;
		INDT_5 << "int32_t delta = 0;" << std::endl;
		INDT_5 << "if( ok ) {" << std::endl;
		INDT_6 << "if( out_idx == FAULT_INDEX ) delta = +fault_delta;" << std::endl;
		INDT_6 << "else if( out_idx == FAULT_INDEX + 1u ) delta = -fault_delta;" << std::endl;
		INDT_6 << "else if( out_idx == FAULT_INDEX + P ) delta = -fault_delta;" << std::endl;
		INDT_6 << "else if( out_idx == FAULT_INDEX + P + 1u ) delta = +fault_delta;" << std::endl;
		INDT_5 << "}" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( delta != 0 ) { " << (use_abyzft_i32_accum ? "acc_scaled_fault" : "acc_scaled") << " += (int64_t)delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( delta != 0 ) { acc32 += delta; acc32_check += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "}" << std::endl;
			if (options.abyzft_gemm) {
				if (use_abyzft_i32_accum)
					INDT_3 << "if( abyzft_scaleAB_mag != 0 ) acc32 = abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled_fault, abyzft_shiftAB);" << std::endl;
				else
					INDT_3 << "if( abyzft_scaleAB_mag != 0 ) acc32 = (int32_t)(abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled, abyzft_shiftAB));" << std::endl;
			}
		INDT_4 << "else if( FAULT_MODEL==2 ) {" << std::endl;
		INDT_5 << "const uint32_t P = " << N << "u;" << std::endl;
		INDT_5 << "int32_t delta = 0;" << std::endl;
		INDT_5 << "for( uint32_t rr=0; rr<FAULT_N; rr++ ) {" << std::endl;
		INDT_6 << "uint32_t base = FAULT_INDEX + rr * FAULT_STRIDE;" << std::endl;
		INDT_6 << "uint32_t base_r = (base / P);" << std::endl;
		INDT_6 << "uint32_t base_c = (base % P);" << std::endl;
		INDT_6 << "bool ok = (base_r + 1u < (uint32_t)" << M << ") && (base_c + 1u < P);" << std::endl;
		INDT_6 << "if( !ok ) continue;" << std::endl;
		INDT_6 << "if( out_idx == base ) delta += +fault_delta;" << std::endl;
		INDT_6 << "else if( out_idx == base + 1u ) delta += -fault_delta;" << std::endl;
		INDT_6 << "else if( out_idx == base + P ) delta += -fault_delta;" << std::endl;
		INDT_6 << "else if( out_idx == base + P + 1u ) delta += +fault_delta;" << std::endl;
		INDT_5 << "}" << std::endl;
		if (options.abyzft_gemm)
			INDT_5 << "if( delta != 0 ) { " << (use_abyzft_i32_accum ? "acc_scaled_fault" : "acc_scaled") << " += (int64_t)delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		else
			INDT_5 << "if( delta != 0 ) { acc32 += delta; acc32_check += delta; FAULT_INJECTED = true; FAULT_INJECTIONS++; }" << std::endl;
		INDT_4 << "}" << std::endl;
		INDT_3 << "}" << std::endl;
			if (options.abyzft_gemm) {
				if (use_abyzft_i32_accum)
					INDT_3 << "if( abyzft_scaleAB_mag != 0 ) acc32 = abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled_fault, abyzft_shiftAB);" << std::endl;
				else
					INDT_3 << "if( abyzft_scaleAB_mag != 0 ) acc32 = (int32_t)(abyzft_scaleAB_sign * ABYZFT_descale_pow2_i64(acc_scaled, abyzft_shiftAB));" << std::endl;
			}

		if (checksum_enabled) {
			if (options.abyzft_gemm) {
				INDT_3 << "acc_row[c] = acc32;" << std::endl;
			}
			else {
				INDT_3 << "acc_row[c] = acc32_check;" << std::endl;
			}
		}

		INDT_3 << float_dtype << " scale = (" << float_dtype << ") (a_scale[0] * b_scale[0]) / y_scale[0];" << std::endl;
		INDT_3 << "double scaled = ((double) acc32) * (double) scale;" << std::endl;
		INDT_3 << "scaled = scaled + (double) y_zero_point[0];" << std::endl;
		INDT_3 << "int32_t q = (int32_t) llround(scaled);" << std::endl;
		INDT_3 << "q = MIN(MAX(q, " << lower << "), " << upper << ");" << std::endl;
		INDT_3 << "Y[r][c] = (" << Y->data_type_str() << ") q;" << std::endl;
		INDT_2 << "}" << std::endl;

		if (checksum_enabled) {
			if (freivalds_enabled || gvfa_enabled) {
				INDT_2 << "for( uint32_t chk=0; chk<" << randomized_checks << "u; chk++ ) {" << std::endl;
				INDT_3 << "uint32_t freivalds_state = (uint32_t)(0x9E3779B9u ^ LAYER_ID ^ r ^ (uint32_t)(chk*0x85EBCA6Bu));" << std::endl;
				if (freivalds_enabled)
					INDT_3 << "uint8_t r_mask[" << N << "];" << std::endl;
				else
					INDT_3 << "float r_vec[" << N << "];" << std::endl;
				if (freivalds_enabled) {
					INDT_3 << "uint32_t r_any = 0;" << std::endl;
					INDT_3 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) { uint32_t bit = ABYZFT_randbit(&freivalds_state); r_mask[cc] = (uint8_t)bit; r_any |= bit; }" << std::endl;
					INDT_3 << "if( !r_any && " << N << "u>0u ) r_mask[0] = 1u;" << std::endl;
				}
				else {
					INDT_3 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) r_vec[cc] = ABYZFT_rand01(&freivalds_state);" << std::endl;
				}
				INDT_3 << "double b_rs[" << K << "];" << std::endl;
				INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) b_rs[i] = 0.0;" << std::endl;
				INDT_3 << "double sumC = 0.0;" << std::endl;
				INDT_3 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) {" << std::endl;
				if (freivalds_enabled) {
					INDT_4 << "if( !r_mask[cc] ) continue;" << std::endl;
					INDT_4 << "sumC += (double)acc_row[cc];" << std::endl;
					INDT_4 << "for( uint32_t i=0; i<" << K << "u; i++ ) b_rs[i] += (double)((int32_t)B[i][cc] - (int32_t)b_zero_point[0]);" << std::endl;
				}
				else {
					INDT_4 << "sumC += (double)acc_row[cc] * (double)r_vec[cc];" << std::endl;
					INDT_4 << "for( uint32_t i=0; i<" << K << "u; i++ ) b_rs[i] += (double)((int32_t)B[i][cc] - (int32_t)b_zero_point[0]) * (double)r_vec[cc];" << std::endl;
				}
				INDT_3 << "}" << std::endl;
				INDT_3 << "double pred = 0.0;" << std::endl;
				INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) pred += (double)((int32_t)A[r][i] - (int32_t)a_zero_point[0]) * b_rs[i];" << std::endl;
				INDT_3 << "double diff = fabs(pred - sumC);" << std::endl;
				INDT_3 << "double tol = " << options.abft_eps << " * (fabs(pred) + 1.0);" << std::endl;
				INDT_3 << "if( diff > tol ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; break; }" << std::endl;
				INDT_2 << "}" << std::endl;
			}
			else {
				INDT_2 << "int64_t sumC = 0;" << std::endl;
				INDT_2 << "int32_t b_s[" << K << "];" << std::endl;
				INDT_2 << "for( uint32_t i=0; i<" << K << "u; i++ ) b_s[i] = 0;" << std::endl;
				INDT_2 << "for( uint32_t cc=0; cc<" << N << "u; cc++ ) {" << std::endl;
				INDT_3 << "sumC += (int64_t)acc_row[cc];" << std::endl;
				INDT_3 << "for( uint32_t i=0; i<" << K << "u; i++ ) b_s[i] += ((int32_t)B[i][cc] - (int32_t)b_zero_point[0]);" << std::endl;
				INDT_2 << "}" << std::endl;
				INDT_2 << "int64_t pred = 0;" << std::endl;
				INDT_2 << "for( uint32_t i=0; i<" << K << "u; i++ ) pred += (int64_t)((int32_t)A[r][i] - (int32_t)a_zero_point[0]) * (int64_t)b_s[i];" << std::endl;
				INDT_2 << "if( sumC != pred ) { TAMPERING_DETECTED = true; TAMPERING_DETECTIONS++; }" << std::endl;
			}
		}

		INDT_1 << "}" << std::endl;
	}

	void resolve(void) override
	{
		name_input(0, "A");
		name_scalar_input(1, "a_scale");
		name_scalar_input(2, "a_zero_point");

		name_input(3, "B");
		name_scalar_input(4, "b_scale");
		name_scalar_input(5, "b_zero_point");

		name_input(6, "y_scale");
		name_scalar_input(7, "y_zero_point");

		Tensor* y_zero_point = get_input_tensor(7);

		Tensor* y = new Tensor;
		y->data_dim = resolve_shape();
		y->data_type = y_zero_point->data_type;
		register_output(y, "Y");
	}
};
} // namespace toC
