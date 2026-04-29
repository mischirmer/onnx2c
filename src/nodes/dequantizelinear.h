/* This file is part of onnx2c.
 *
 * DequantizeLinear node.
 *
 */

#include "node.h"

namespace toC {

class DequantizeLinear : public Node {
	public:
	DequantizeLinear()
	{
		op_name = "DequantizeLinear";
		axis = 1;
	}

	// Attributes
	int axis;

	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;
};

void DequantizeLinear::parseAttributes(onnx::NodeProto& node)
{
	for (const auto& a : node.attribute()) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if (a.name() == "axis")
			axis = parse_attribute_int(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node DequantizeLinear/" << onnx_name << std::endl;
	}
}

void DequantizeLinear::resolve(void)
{
	Tensor* x = get_input_tensor(0);
	Tensor* x_scale = get_input_tensor(1);
	name_input(0, "x");
	name_input(1, "x_scale");

	if (axis < 0) {
		axis += x->data_dim.size();
	}

	if (get_number_of_inputs() == 3) {
		name_input(2, "x_zero_point");
	}

	Tensor* t = new Tensor;
	t->data_dim = x->data_dim;
	t->data_type = x_scale->data_type;
	register_output(t, "y");
}

void DequantizeLinear::print(std::ostream& dst) const
{
	INDT_1 << "/* DequantizeLinear */" << std::endl;

	Tensor* x = get_input_tensor(0);
	Tensor* x_scale = get_input_tensor(1);

	std::string index;
	for (unsigned loop_axis = 0; loop_axis < x->rank(); loop_axis++) {
		std::string name = "i" + std::to_string(loop_axis);
		INDT_1 << "for (unsigned " << name << " = 0; " << name << " < " << x->data_dim[loop_axis] << "; " << name << "++)" << std::endl;

		index += "[" + name + "]";
	}

	std::string param_index;
	// ONNX allows x_scale/x_zero_point to be a 1-element tensor (shape [1]).
	// Treat that case like a scalar to avoid emitting out-of-range indices
	// (e.g. default axis=1 for rank-1 inputs) and undefined loop variables.
	if (x_scale->is_scalar() || x_scale->data_num_elem() == 1 || axis >= (int)x->rank()) {
		param_index = "[0]";
	}
	else {
		param_index = "[i" + std::to_string(axis) + "]";
	}

	INDT_1 << "{" << std::endl;
	INDT_2 << "y" << index << " = (x" << index;
	if (get_number_of_inputs() == 3) {
		Tensor* x_zero_point = get_input_tensor(2);
		if (x_zero_point->is_scalar() || x_zero_point->data_num_elem() == 1 || axis >= (int)x->rank())
			dst << " - x_zero_point[0]";
		else
			dst << " - x_zero_point" << param_index;
	}
	dst << ") * x_scale" << param_index << ";" << std::endl;
	INDT_1 << "}" << std::endl;
}

} // namespace toC
