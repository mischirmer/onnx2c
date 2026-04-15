/* This file is part of onnx2c.
 *
 * MatMulInteger
 * Matrix multiplication with integers.
 * In contrast to MatMul (which allows floats only)
 * MatMulInteger takes a input zero-point bias term
 * which is useful for quantized networks.
 */

#include "abstractmatmul.h"

namespace toC {

class MatMulInteger : public AbstractMatMul {
	public:
	MatMulInteger()
	{
		op_name = "MatMulInteger";
		b_zero_point_per_output_channel = false;
	}
	bool b_zero_point_per_output_channel;

	void print_multiply_accumulate(std::ostream& dst,
	                               const std::string& y_idx,
	                               const std::string& a_idx,
	                               const std::string& b_idx) const override
	{
		const char* bzp_idx = b_zero_point_per_output_channel ? "j" : "0";
		INDT_4 << y_idx << " += (" << a_idx << " - a_zero_point[0]) * ("
		       << b_idx << " - b_zero_point[" << bzp_idx << "]);" << std::endl;
	}

	void resolve(void) override
	{
		auto is_const_splatted = [](const Tensor* t) -> bool {
			if (!t->isConst || t->data_num_elem() <= 1)
				return true;
			const int n = t->data_num_elem();
			switch (t->data_type) {
				case onnx::TensorProto_DataType_FLOAT: {
					const float* p = (const float*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_DOUBLE: {
					const double* p = (const double*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_INT8: {
					const int8_t* p = (const int8_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_UINT8: {
					const uint8_t* p = (const uint8_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_INT16: {
					const int16_t* p = (const int16_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_UINT16: {
					const uint16_t* p = (const uint16_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_INT32: {
					const int32_t* p = (const int32_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_UINT32: {
					const uint32_t* p = (const uint32_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_INT64: {
					const int64_t* p = (const int64_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				case onnx::TensorProto_DataType_UINT64: {
					const uint64_t* p = (const uint64_t*)t->data_buffer;
					for (int i = 1; i < n; i++)
						if (p[i] != p[0]) return false;
					return true;
				}
				default:
					return false;
			}
		};

		auto canonicalize_scalar_like = [&](Tensor* t) {
			if (t->is_scalar() || (t->data_dim.size() == 1 && t->data_dim[0] == 1))
				return;
			if (t->data_num_elem() == 1) {
				t->data_dim = {1};
				return;
			}
			// Accept constant broadcasted/splatted tensors as scalar-like.
			if (is_const_splatted(t))
				t->data_dim = {1};
		};

		name_input(0, "A");
		name_input(1, "B");

		if (get_number_of_inputs() > 2) {
			Tensor* a_zp = get_input_tensor(2);
			canonicalize_scalar_like(a_zp);
			name_input(2, "a_zero_point");
			if (!(a_zp->is_scalar() || (a_zp->data_dim.size() == 1 && a_zp->data_dim[0] == 1))) {
				ERROR("a_zero_point must be scalar (rank-0) or [1], got name=" << a_zp->name
				      << " shape={" << a_zp->str_dimensions() << "} rank=" << a_zp->rank()
				      << " elems=" << a_zp->data_num_elem() << " dtype=" << a_zp->data_type
				      << " const=" << a_zp->isConst);
			}
		}

		if (get_number_of_inputs() > 3) {
			Tensor* b_zp = get_input_tensor(3);
			canonicalize_scalar_like(b_zp);
			name_input(3, "b_zero_point");
			b_zero_point_per_output_channel = false;
			if (b_zp->rank() == 1 && get_b()->rank() > 1 &&
			    b_zp->data_dim[0] == get_b()->data_dim[get_b()->rank() - 1]) {
				b_zero_point_per_output_channel = true;
			}
			else if (!(b_zp->is_scalar() || (b_zp->data_dim.size() == 1 && b_zp->data_dim[0] == 1))) {
				ERROR("b_zero_point must be scalar (rank-0) or [1], got name=" << b_zp->name
				      << " shape={" << b_zp->str_dimensions() << "} rank=" << b_zp->rank()
				      << " elems=" << b_zp->data_num_elem() << " dtype=" << b_zp->data_type
				      << " const=" << b_zp->isConst);
			}
		}

		Tensor* y = new Tensor;
		y->data_dim = resolve_shape();
		y->data_type = onnx::TensorProto_DataType_INT32;
		register_output(y, "Y");
	}
};

} // namespace toC
