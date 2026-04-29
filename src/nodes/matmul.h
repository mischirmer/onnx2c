/* This file is part of onnx2c.
 *
 * MatMul node.
 *
 */

#include "abstractmatmul.h"
#include "node.h"

namespace toC {

class MatMul : public AbstractMatMul {
	public:
	MatMul()
	{
		op_name = "MatMul";
	}

	virtual void resolve(void) override;
	void print(std::ostream& dst) const override;
	void print_multiply_accumulate(std::ostream& dst,
	                               const std::string& y_idx,
	                               const std::string& a_idx,
	                               const std::string& b_idx) const override;
};

} // namespace toC
