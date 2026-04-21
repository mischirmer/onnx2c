#pragma once

#include "../node.h"

namespace toC {

class Im2Col : public Node {
public:
    Im2Col();
    std::string c_name(void) const;
    std::string c_output_decl(void);
    void print_node(std::ostream& dst) const;
    void resolve_datatypes(void);
    void print(std::ostream& destination) const override;

    int64_t batch = 1;
    int64_t in_ch = 1;
    int64_t out_ch = 1;
    int64_t in_h = 1;
    int64_t in_w = 1;
    int64_t kernel_h = 1;
    int64_t kernel_w = 1;
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    int64_t dilation_h = 1;
    int64_t dilation_w = 1;
    int64_t out_h = 1;
    int64_t out_w = 1;
    int64_t group = 1;
    bool has_bias = false;
};

}