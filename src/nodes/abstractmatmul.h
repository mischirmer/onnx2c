/* This file is part of onnx2c.
 *
 * AbstractMatMul
 */

#pragma once

#include "node.h"

namespace toC {

class AbstractMatMul : public Node {
	public:
	using Node::Node;

	virtual void print(std::ostream& dst) const override;

	virtual void print_initialize(std::ostream& dst, const std::string& y_idx) const
	{
		INDT_3 << y_idx << " = 0;" << std::endl;
	}

	virtual void print_finalize(std::ostream& dst, const std::string& y_idx) const
	{
	}

	virtual void print_multiply_accumulate(std::ostream& dst,
	                                       const std::string& y_idx,
	                                       const std::string& a_idx,
	                                       const std::string& b_idx) const = 0;

	virtual Tensor* get_a() const { return get_input_tensor(0); }
	virtual Tensor* get_b() const { return get_input_tensor(1); }

	std::vector<int> resolve_shape() const;
};

} // namespace toC
