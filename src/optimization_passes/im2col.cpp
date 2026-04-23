/* This file is part of onnx2c.
 *
 * Im2Col optimization pass
 * Transforms Conv operations into fused Im2Col node for better performance.
 * Handles: Conv, depthwise (group>1), and pointwise (1x1) convolutions.
 */

#include "graph.h"
#include "nodes/spatialfilter.h"
#include "nodes/im2col.h"
#include "options.h"

#include <cstdint>
#include <algorithm>

using namespace toC;

static bool is_conv_like_node(Node* n)
{
	return n->op_name == "Conv" ||
	       n->op_name == "ConvInteger" ||
	       n->op_name == "QLinearConv";
}

static void replace_consumer(Tensor* tensor, Node* old_consumer, Node* new_consumer)
{
	for (auto& consumer : tensor->consumers) {
		if (consumer == old_consumer) {
			consumer = new_consumer;
			return;
		}
	}
	for (auto consumer : tensor->consumers) {
		if (consumer == new_consumer)
			return;
	}
	tensor->consumers.push_back(new_consumer);
}

static bool is_arithmetic_type(onnx::TensorProto_DataType t)
{
	return t == onnx::TensorProto_DataType_FLOAT ||
	       t == onnx::TensorProto_DataType_DOUBLE ||
	       t == onnx::TensorProto_DataType_FLOAT16 ||
	       t == onnx::TensorProto_DataType_BFLOAT16 ||
	       t == onnx::TensorProto_DataType_INT8 ||
	       t == onnx::TensorProto_DataType_UINT8 ||
	       t == onnx::TensorProto_DataType_INT16 ||
	       t == onnx::TensorProto_DataType_UINT16 ||
	       t == onnx::TensorProto_DataType_INT32 ||
	       t == onnx::TensorProto_DataType_UINT32 ||
	       t == onnx::TensorProto_DataType_INT64 ||
	       t == onnx::TensorProto_DataType_UINT64;
}

static bool can_be_im2col(SpatialFilter* conv, Tensor* X, Tensor* W, Tensor* Y)
{
	if (X->rank() != 4 || W->rank() != 4 || Y->rank() != 4)
		return false;

	if (!is_arithmetic_type(X->data_type) ||
	    !is_arithmetic_type(W->data_type) ||
	    !is_arithmetic_type(Y->data_type))
		return false;

	if (conv->group <= 0)
		return false;

	if (X->data_dim[1] % conv->group != 0 || W->data_dim[0] % conv->group != 0)
		return false;

	if (W->data_dim[1] != X->data_dim[1] / conv->group)
		return false;

	return true;
}

static bool has_non_unit_dilation(SpatialFilter* conv)
{
	for (auto dilation : conv->dilations) {
		if (dilation != 1)
			return true;
	}
	return false;
}

static bool should_use_im2col_heuristic(SpatialFilter* conv, Tensor* X, Tensor* W, Tensor* Y)
{
	int64_t batch = X->data_dim[0];
	int64_t in_ch = X->data_dim[1];
	int64_t in_h = X->data_dim[2];
	int64_t in_w = X->data_dim[3];
	int64_t out_ch = Y->data_dim[1];
	int64_t out_h = Y->data_dim[2];
	int64_t out_w = Y->data_dim[3];
	int64_t kernel_h = W->data_dim[2];
	int64_t kernel_w = W->data_dim[3];
	int64_t group = conv->group;
	int64_t in_ch_per_group = in_ch / group;

	int64_t ifm_area = in_h * in_w;
	int64_t output_cells = batch * out_h * out_w * out_ch;
	int64_t kernel_area = kernel_h * kernel_w;
	int64_t lowered_elems = batch * out_h * out_w * in_ch_per_group * kernel_area;

	// The Arm Compute Library CPU heuristic routes non-unit dilation to
	// Im2Col+GEMM, and 1x1 kernels to GEMM. onnx2c only selects between
	// direct loops and the fused im2col loops, but these are still strong
	// indicators that the im2col-style loop order is useful.
	if (has_non_unit_dilation(conv))
		return true;

	if (kernel_h == 1 && kernel_w == 1)
		return true;

	// Avoid forcing the transformation on very large lowered matrices by
	// default. In a classical im2col implementation this corresponds to a
	// large temporary workspace; for onnx2c's fused implementation it is still
	// a useful proxy for cases where direct convolution may be preferable.
	const int64_t large_lowered_matrix_elems = 2 * 1024 * 1024;
	if (lowered_elems > large_lowered_matrix_elems)
		return false;

	// Small feature maps are generally good candidates for GEMM-style loop
	// orders because the overhead is low and the reuse is dense.
	if (ifm_area <= 64)
		return true;

	// Depthwise/grouped convolutions have a small per-output reduction and tend
	// to benefit from avoiding the generic direct grouped loop structure.
	if (group > 1)
		return true;

	// For regular kernels, require a modest amount of arithmetic work before
	// choosing im2col by default.
	return kernel_area > 1 && output_cells * in_ch_per_group * kernel_area >= 4096;
}

static Im2Col::ArithmeticMode arithmetic_mode_for(Node* n)
{
	if (n->op_name == "ConvInteger")
		return Im2Col::ConvInteger;
	if (n->op_name == "QLinearConv")
		return Im2Col::QLinearConv;
	return Im2Col::Conv;
}

void Graph::im2col(void)
{
	LOG(DEBUG) << "Optimisation pass: im2col" << std::endl;

	std::vector<Node*> candidates;

	LOG(TRACE) << "im2col pass: entered" << std::endl;

	int nodes_before = nodes.size();
	int tensors_before = tensors.size();
	LOG(TRACE) << "im2col pass: nodes=" << nodes_before << " tensors=" << tensors_before << std::endl;

	for (auto n : nodes) {
		if (!is_conv_like_node(n))
			continue;

		LOG(DEBUG) << "Considering im2col for node: " << n->onnx_name << " (op: " << n->op_name << ")" << std::endl;

		SpatialFilter* conv = dynamic_cast<SpatialFilter*>(n);
		if (!conv) {
			LOG(WARNING) << "  Skipping: not a SpatialFilter" << std::endl;
			continue;
		}

		Tensor* X = const_cast<Tensor*>(conv->get_X());
		Tensor* W = const_cast<Tensor*>(conv->get_W());
		Tensor* Y = conv->get_output_tensor(0);
		LOG(TRACE) << "  Input tensor: " << X->name << " rank=" << X->rank() << std::endl;
		if (!can_be_im2col(conv, X, W, Y)) {
			LOG(WARNING) << "  Skipping: unsupported Conv for im2col" << std::endl;
			continue;
		}

		if (options.opt_im2col_mode == im2col_mode::HEURISTIC &&
		    !should_use_im2col_heuristic(conv, X, W, Y)) {
			LOG(DEBUG) << "  Skipping: im2col heuristic selected baseline Conv" << std::endl;
			continue;
		}

		candidates.push_back(n);
	}

	for (auto n : candidates) {
		SpatialFilter* conv = dynamic_cast<SpatialFilter*>(n);

		Tensor* X = const_cast<Tensor*>(conv->get_X());
		Tensor* W = const_cast<Tensor*>(conv->get_W());
		Tensor* Y = conv->get_output_tensor(0);

		Im2Col* im2col = new Im2Col;
		im2col->onnx_name = conv->onnx_name + "_im2col";
		im2col->isResolved = true;
		im2col->arithmetic_mode = arithmetic_mode_for(n);

		im2col->batch = X->data_dim[0];
		im2col->in_ch = X->data_dim[1];
		im2col->in_h = X->data_dim[2];
		im2col->in_w = X->data_dim[3];

		im2col->out_ch = W->data_dim[0];
		im2col->kernel_h = W->data_dim[2];
		im2col->kernel_w = W->data_dim[3];

		im2col->out_h = Y->data_dim[2];
		im2col->out_w = Y->data_dim[3];

		im2col->group = conv->group;

		if (conv->strides.size() >= 2) {
			im2col->stride_h = conv->strides[0];
			im2col->stride_w = conv->strides[1];
		}
		if (conv->pads.size() >= 4) {
			im2col->pad_h = conv->pads[0];
			im2col->pad_w = conv->pads[1];
		}
		if (conv->dilations.size() >= 2) {
			im2col->dilation_h = conv->dilations[0];
			im2col->dilation_w = conv->dilations[1];
		}

		if (im2col->arithmetic_mode == Im2Col::Conv) {
			im2col->has_bias = (conv->get_number_of_inputs() >= 3);
			im2col->register_input(X, "x");
			im2col->register_input(W, "w");
			if (im2col->has_bias) {
				Tensor* B = conv->get_input_tensor(2);
				im2col->register_input(B, "bias");
				replace_consumer(B, n, im2col);
			}
		}
		else if (im2col->arithmetic_mode == Im2Col::ConvInteger) {
			im2col->has_x_zero_point = (conv->get_number_of_inputs() >= 3);
			im2col->has_w_zero_point = (conv->get_number_of_inputs() >= 4);
			im2col->register_input(X, "x");
			im2col->register_input(W, "w");
			if (im2col->has_x_zero_point) {
				Tensor* X_zero_point = conv->get_input_tensor(2);
				im2col->register_input(X_zero_point, "x_zero_point");
				replace_consumer(X_zero_point, n, im2col);
			}
			if (im2col->has_w_zero_point) {
				Tensor* W_zero_point = conv->get_input_tensor(3);
				im2col->register_input(W_zero_point, "w_zero_point");
				replace_consumer(W_zero_point, n, im2col);
			}
		}
		else {
			im2col->has_bias = (conv->get_number_of_inputs() >= 9);
			im2col->register_input(X, "x");
			im2col->register_input(conv->get_input_tensor(1), "x_scale");
			im2col->register_input(conv->get_input_tensor(2), "x_zero_point");
			im2col->register_input(W, "w");
			im2col->register_input(conv->get_input_tensor(4), "w_scale");
			im2col->register_input(conv->get_input_tensor(5), "w_zero_point");
			im2col->register_input(conv->get_input_tensor(6), "y_scale");
			im2col->register_input(conv->get_input_tensor(7), "y_zero_point");
			for (unsigned i = 1; i <= 7; i++)
				replace_consumer(conv->get_input_tensor(i), n, im2col);
			if (im2col->has_bias) {
				Tensor* B = conv->get_input_tensor(8);
				im2col->register_input(B, "bias");
				replace_consumer(B, n, im2col);
			}
		}
		im2col->register_output(Y, "y");

		replace_consumer(X, n, im2col);
		replace_consumer(W, n, im2col);

		auto it = std::find(nodes.begin(), nodes.end(), n);
		if (it != nodes.end()) {
			*it = im2col;
		}
		std::string transformed_node_name = n->onnx_name;
		delete n;

		LOG(TRACE) << "Transform complete for " << transformed_node_name << std::endl;
	}

	int nodes_after = nodes.size();
	int tensors_after = tensors.size();
	LOG(TRACE) << "im2col pass: after: nodes=" << nodes_after << " tensors=" << tensors_after << std::endl;

	LOG(TRACE) << "im2col pass finished" << std::endl;
}
