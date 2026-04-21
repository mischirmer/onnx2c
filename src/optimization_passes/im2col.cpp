/* This file is part of onnx2c.
 *
 * Im2Col optimization pass (DISABLED)
 * Transforms Conv operations into fused Im2Col node for better performance.
 * Handles: Conv, depthwise (group>1), and pointwise (1x1) convolutions.
 *
 * Currently disabled due to bugs in the code generation.
 */

#include "graph.h"
#include "nodes/spatialfilter.h"
#include "nodes/im2col.h"

#include <cstdint>
#include <algorithm>

using namespace toC;

static bool is_conv_like_node(Node* n)
{
	return n->op_name == "Conv" ||
	       n->op_name == "ConvInteger" ||
	       n->op_name == "QLinearConv";
}

static bool can_be_im2col(SpatialFilter* conv)
{
	if (conv->dilations.size() > 0) {
		for (size_t i = 0; i < conv->dilations.size(); i++) {
			if (conv->dilations[i] != 1)
				return false;
		}
	}
	return true;
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

		if (!can_be_im2col(conv)) {
			LOG(WARNING) << "  Skipping: unsupported dilations" << std::endl;
			continue;
		}

		Tensor* X = conv->get_input_tensor(0);
		LOG(TRACE) << "  Input tensor: " << X->name << " rank=" << X->rank() << std::endl;
		if (X->rank() != 4) {
			LOG(WARNING) << "  Skipping: only 2D convolutions supported" << std::endl;
			continue;
		}

		candidates.push_back(n);
	}

	for (auto n : candidates) {
		SpatialFilter* conv = dynamic_cast<SpatialFilter*>(n);

		Tensor* X = conv->get_input_tensor(0);
		Tensor* W = conv->get_input_tensor(1);
		Tensor* Y = conv->get_output_tensor(0);

		Im2Col* im2col = new Im2Col;
		im2col->onnx_name = conv->onnx_name + "_im2col";
		im2col->isResolved = true;

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
			im2col->pad_w = conv->pads[2];
		}
		if (conv->dilations.size() >= 2) {
			im2col->dilation_h = conv->dilations[0];
			im2col->dilation_w = conv->dilations[1];
		}

		im2col->has_bias = (conv->get_number_of_inputs() >= 3);

		Tensor* output = new Tensor;
		output->name = im2col->onnx_name + "_output";
		output->data_type = Y->data_type;
		output->data_dim = Y->data_dim;
		output->isConst = false;
		output->initialize = false;
		output->generate = true;

		nodes.push_back(im2col);
		tensors.push_back(output);

		im2col->register_input(X, "x");
		im2col->register_input(W, "w");
		if (im2col->has_bias) {
			Tensor* B = conv->get_input_tensor(2);
			im2col->register_input(B, "b");
		}
		im2col->register_output(output, "y");

		X->consumers.push_back(im2col);
		W->consumers.push_back(im2col);

		for (auto consumer : Y->consumers) {
			consumer->replace_input(Y, output);
		}

		auto it = std::find(nodes.begin(), nodes.end(), n);
		if (it != nodes.end()) {
			nodes.erase(it);
		}
		delete n;

		LOG(TRACE) << "Transform complete for " << n->onnx_name << std::endl;
	}

	int nodes_after = nodes.size();
	int tensors_after = tensors.size();
	LOG(TRACE) << "im2col pass: after: nodes=" << nodes_after << " tensors=" << tensors_after << std::endl;

	LOG(TRACE) << "im2col pass finished" << std::endl;
}