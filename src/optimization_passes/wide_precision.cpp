#include "graph.h"
#include "options.h"

#include <cstdlib>
#include <cstring>

using namespace toC;

static bool is_promotable_int8_weight(const Tensor* t)
{
	return t && t->isConst && t->data_buffer &&
	       (t->data_type == onnx::TensorProto_DataType_INT8 ||
	        t->data_type == onnx::TensorProto_DataType_UINT8);
}

static bool is_weight_input(Node* n, Tensor* t)
{
	if (!n || !t)
		return false;

	// Quantized operators
	if (n->op_name == "QLinearConv" && n->get_number_of_inputs() > 3)
		return n->get_input_tensor(3) == t;
	if (n->op_name == "QLinearMatMul" && n->get_number_of_inputs() > 3)
		return n->get_input_tensor(3) == t;
	if (n->op_name == "ConvInteger" && n->get_number_of_inputs() > 1)
		return n->get_input_tensor(1) == t;
	if (n->op_name == "MatMulInteger" && n->get_number_of_inputs() > 1)
		return n->get_input_tensor(1) == t;
	if (n->op_name == "QGemm" && n->get_number_of_inputs() > 3)
		return n->get_input_tensor(3) == t;

	// Regular operators with int8 weights
	if (n->op_name == "MatMul" && n->get_number_of_inputs() > 1)
		return n->get_input_tensor(1) == t;
	if (n->op_name == "Conv" && n->get_number_of_inputs() > 1)
		return n->get_input_tensor(1) == t;
	if (n->op_name == "Gemm" && n->get_number_of_inputs() > 1)
		return n->get_input_tensor(1) == t;

	return false;
}

static bool is_used_as_weight(Tensor* t)
{
	for (auto* consumer : t->consumers) {
		if (is_weight_input(consumer, t))
			return true;
	}
	return false;
}

static void promote_int8_weight(Tensor* t)
{
	const int elems = t->data_num_elem();
	if (t->data_type == onnx::TensorProto_DataType_INT8) {
		int16_t* widened = static_cast<int16_t*>(malloc(static_cast<size_t>(elems) * sizeof(int16_t)));
		if (!widened)
			ERROR("memory allocation failed while widening tensor " << t->name);
		const int8_t* src = static_cast<const int8_t*>(t->data_buffer);
		for (int i = 0; i < elems; i++)
			widened[i] = static_cast<int16_t>(src[i]);
		free(t->data_buffer);
		t->data_buffer = widened;
		t->data_type = onnx::TensorProto_DataType_INT16;
		return;
	}

	uint16_t* widened = static_cast<uint16_t*>(malloc(static_cast<size_t>(elems) * sizeof(uint16_t)));
	if (!widened)
		ERROR("memory allocation failed while widening tensor " << t->name);
	const uint8_t* src = static_cast<const uint8_t*>(t->data_buffer);
	for (int i = 0; i < elems; i++)
		widened[i] = static_cast<uint16_t>(src[i]);
	free(t->data_buffer);
	t->data_buffer = widened;
	t->data_type = onnx::TensorProto_DataType_UINT16;
}

void Graph::wide_precision(void)
{
	LOG(DEBUG) << "Optimisation pass: wide_precision" << std::endl;

	for (auto* t : tensors) {
		if (!is_promotable_int8_weight(t) || !is_used_as_weight(t))
			continue;

		const auto old_type = t->data_type;
		promote_int8_weight(t);
		LOG(DEBUG) << "Promoted weight tensor " << t->name
		           << " from " << (old_type == onnx::TensorProto_DataType_INT8 ? "int8" : "uint8")
		           << " to " << t->data_type_str() << std::endl;
	}
}
