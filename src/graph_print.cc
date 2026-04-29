/* This file is part of onnx2c
 */

#include "error.h"
#include "graph.h"
#include "options.h"
#include "timestamp.h"
#include "util.h"

#include <iostream>

using namespace toC;

void Graph::print_header(std::ostream& dst, const std::string& interface_func_name)
{
	print_file_frontmatter(dst);
}

void Graph::print_source(std::ostream& dst, const std::string& interface_func_name)
{
	print_file_frontmatter(dst);
	dst << std::endl;
	print_includes(dst);
	dst << std::endl;
	// ABFT status globals (updated by generated ABFT checks when enabled).
	dst << "volatile bool TAMPERING_DETECTED = false;" << std::endl;
	dst << "uint32_t TAMPERING_DETECTIONS = 0;" << std::endl;
	// Fault injection controls (used by generated kernels when enabled).
	dst << "volatile bool FAULT_ENABLED = false;" << std::endl;
	dst << "uint32_t FAULT_MODEL = 0; /* 0=single_point, 1=trivial, 2=checkered */" << std::endl;
	dst << "uint32_t FAULT_LAYER_ID = 0;" << std::endl;
	dst << "uint32_t FAULT_INDEX = 0;" << std::endl;
	dst << "float FAULT_VALUE = 0.0f;" << std::endl;
	dst << "uint32_t FAULT_N = 12; /* used by some fault models */" << std::endl;
	dst << "uint32_t FAULT_STRIDE = 2; /* used by some fault models */" << std::endl;
	dst << "volatile bool FAULT_INJECTED = false;" << std::endl;
	dst << "uint32_t FAULT_INJECTIONS = 0;" << std::endl;
	dst << std::endl;

	// Deterministic PRNG helpers (used by AByzFT / Freivalds / GVFA).
	// Only emit the helpers that are actually needed by the enabled mechanism(s),
	// to keep generated code minimal and avoid confusion in analyses.
	if (options.abyzft_gemm || options.freivalds_gemm || options.gvfa_gemm) {
		dst << "static inline uint32_t ABYZFT_xorshift32(uint32_t* s) {" << std::endl;
		dst << "\tuint32_t x = *s;" << std::endl;
		dst << "\tx ^= x << 13;" << std::endl;
		dst << "\tx ^= x >> 17;" << std::endl;
		dst << "\tx ^= x << 5;" << std::endl;
		dst << "\t*s = x;" << std::endl;
		dst << "\treturn x;" << std::endl;
		dst << "}" << std::endl;
		if (options.abyzft_gemm || options.gvfa_gemm) {
			dst << "static inline float ABYZFT_rand01(uint32_t* s) {" << std::endl;
			dst << "\t// [0,1) using 24 bits" << std::endl;
			dst << "\treturn (float)(ABYZFT_xorshift32(s) & 0x00FFFFFFu) / 16777216.0f;" << std::endl;
			dst << "}" << std::endl;
		}
		if (options.abyzft_gemm) {
			dst << "#ifndef ABYZFT_U8_SCALE_COUNT" << std::endl;
			dst << "#define ABYZFT_U8_SCALE_COUNT 6u" << std::endl;
			dst << "#endif" << std::endl;
			dst << "#ifndef ABYZFT_U8_SCALES" << std::endl;
			dst << "#define ABYZFT_U8_SCALES {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f}" << std::endl;
			dst << "#endif" << std::endl;
			dst << "#ifndef ABYZFT_S8_SCALE_COUNT" << std::endl;
			dst << "#define ABYZFT_S8_SCALE_COUNT 12u" << std::endl;
			dst << "#endif" << std::endl;
			dst << "#ifndef ABYZFT_S8_SCALES" << std::endl;
			dst << "#define ABYZFT_S8_SCALES {-64.0f, -32.0f, -16.0f, -8.0f, -4.0f, -2.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f}" << std::endl;
			dst << "#endif" << std::endl;
			dst << "static inline float ABYZFT_pick_scale_u8(uint32_t* s) {" << std::endl;
			dst << "\tstatic const float k[] = ABYZFT_U8_SCALES;" << std::endl;
			dst << "\treturn k[ABYZFT_xorshift32(s) % ABYZFT_U8_SCALE_COUNT];" << std::endl;
			dst << "}" << std::endl;
			dst << "static inline float ABYZFT_pick_scale_s8(uint32_t* s) {" << std::endl;
			dst << "\tstatic const float k[] = ABYZFT_S8_SCALES;" << std::endl;
			dst << "\treturn k[ABYZFT_xorshift32(s) % ABYZFT_S8_SCALE_COUNT];" << std::endl;
			dst << "}" << std::endl;
			dst << "static inline uint32_t ABYZFT_pow2_shift_u64(uint64_t v) {" << std::endl;
			dst << "\treturn v ? (uint32_t)__builtin_ctzll((unsigned long long)v) : 0u;" << std::endl;
			dst << "}" << std::endl;
			dst << "static inline int64_t ABYZFT_descale_pow2_i64(int64_t v, uint32_t shift) {" << std::endl;
			dst << "\tif( shift == 0u ) return v;" << std::endl;
			dst << "\tconst int64_t bias = (v < 0) ? (((int64_t)1 << shift) - 1) : 0;" << std::endl;
			dst << "\treturn (v + bias) >> shift;" << std::endl;
			dst << "}" << std::endl;
			dst << "static inline int32_t ABYZFT_descale_pow2_i32(int32_t v, uint32_t shift) {" << std::endl;
			dst << "\tif( shift == 0u ) return v;" << std::endl;
			dst << "\tconst int32_t bias = (v < 0) ? (((int32_t)1 << shift) - 1) : 0;" << std::endl;
			dst << "\treturn (v + bias) >> shift;" << std::endl;
			dst << "}" << std::endl;
		}
		if (options.freivalds_gemm) {
			dst << "static inline uint32_t ABYZFT_randbit(uint32_t* s) {" << std::endl;
			dst << "\treturn ABYZFT_xorshift32(s) & 1u;" << std::endl;
			dst << "}" << std::endl;
		}
		if (options.gvfa_gemm) {
			dst << "static inline float ABYZFT_randn(uint32_t* s) {" << std::endl;
			dst << "\t// Box-Muller transform for N(0,1)" << std::endl;
			dst << "\tfloat u1 = ABYZFT_rand01(s);" << std::endl;
			dst << "\tfloat u2 = ABYZFT_rand01(s);" << std::endl;
			dst << "\tif( u1 < 1.0e-7f ) u1 = 1.0e-7f;" << std::endl;
			dst << "\treturn sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);" << std::endl;
			dst << "}" << std::endl;
		}
		dst << std::endl;
	}

	// Minimal metadata for fault-injection sweeps (selected compute-heavy layers).
	// The runtime can iterate these to run per-layer sweeps without parsing C/ONNX.
	{
		std::vector<const Node*> sweep_nodes;
		for (auto* n : nodes) {
			if (!n) continue;
			n->sweep_layer_id = 0;
			// Sweep only compute-heavy layers where faults are meaningful:
			// - Conv / depthwise conv: Conv, QLinearConv, ConvInteger
			// - Dense: Gemm, QGemm, MatMul, QLinearMatMul
			if (n->op_name == "Conv" ||
			    n->op_name == "QLinearConv" ||
			    n->op_name == "ConvInteger" ||
			    n->op_name == "Gemm" ||
			    n->op_name == "QGemm" ||
			    n->op_name == "MatMul" ||
			    n->op_name == "QLinearMatMul") {
				n->sweep_layer_id = (uint32_t)(sweep_nodes.size() + 1u);
				sweep_nodes.push_back(n);
			}
		}
		dst << "const uint32_t SWEEP_LAYER_COUNT = " << sweep_nodes.size() << "u;" << std::endl;
		dst << "const uint32_t SWEEP_LAYER_IDS[" << sweep_nodes.size() << "] = {";
		for (size_t i = 0; i < sweep_nodes.size(); i++) {
			if (i) dst << ", ";
			dst << (uint32_t)(i + 1) << "u";
		}
		dst << "};" << std::endl;
		dst << "const char* SWEEP_LAYER_OPS[" << sweep_nodes.size() << "] = {";
		for (size_t i = 0; i < sweep_nodes.size(); i++) {
			if (i) dst << ", ";
			dst << "\"" << sweep_nodes[i]->op_name << "\"";
		}
		dst << "};" << std::endl;
		dst << "const uint64_t SWEEP_LAYER_OUT_ELEMS[" << sweep_nodes.size() << "] = {";
		for (size_t i = 0; i < sweep_nodes.size(); i++) {
			if (i) dst << ", ";
			uint64_t elems = 1;
			const Tensor* out0 = sweep_nodes[i]->get_output_tensor(0);
			if (!out0) {
				elems = 0;
			}
			else {
				for (int d : out0->data_dim) {
					if (d <= 0) { elems = 0; break; }
					elems *= (uint64_t)d;
				}
			}
			dst << elems << "ull";
		}
		dst << "};" << std::endl;
	}
	dst << std::endl;

	print_global_tensors(dst);
	dst << std::endl;
	print_functions(dst);
	dst << std::endl;
	print_interface_function(dst, /*print_definition=*/true, interface_func_name);
}

void Graph::print_initialization(std::ostream& dst)
{
	print_file_frontmatter(dst);
	dst << std::endl;
	print_includes(dst);
	dst << std::endl;

	LOG(TRACE) << "printing initializers" << std::endl;
	for (auto t : tensors) {
		LOG(TRACE) << "\t" << t->print_trace_dump() << std::endl;
		if (t->union_no < 0 && t->generate && t->initialize)
			print_tensor(t, dst);
	}
	LOG(TRACE) << "(done printing initializers)" << std::endl;
}

void Graph::print_file_frontmatter(std::ostream& dst)
{
	// TODO: Validate inputs, especially check for newlines

	dst << "// This file is computer-generated by onnx2c " << std::endl;
	dst << "// Command Line:";
	for (const std::string& arg : options.command_line_args) {
		dst << " " << arg;
	}
	dst << std::endl;

	dst << std::endl;

	dst << "// onnx2c" << std::endl;
	dst << "// Git Branch: " << git_branch_str << std::endl;
	dst << "// Git Commit: " << git_short_hash_str << std::endl;

	dst << std::endl;

	dst << "// ONNX Model" << std::endl;
	dst << "// Produced By: " << model.producer_name();
	dst << ", version " << model.producer_version() << std::endl;
	dst << "// ONNX IR version: " << onnx_ir_version() << std::endl;

	if (model.doc_string().size() > 0) {
		dst << "// Model documentation:" << std::endl;

		size_t start = 0;
		size_t line_end = model.doc_string().find("\n");
		while (line_end != std::string::npos) {
			dst << "// " << model.doc_string().substr(start, line_end - start) << std::endl;
			start = line_end + 1;
			line_end = model.doc_string().find("\n", start);
		}
		dst << "// " << model.doc_string().substr(start) << std::endl;
	}
}

void Graph::print_tensor(const Tensor* t, std::ostream& dst)
{
	if (t->generate == false)
		return;
	if (t->name == "")
		return;
	// This case has been seen in the wild. Not sure why it happens
	if (t->data_dim.size() == 1 && t->data_dim[0] == 0) {
		LOG(WARNING) << "Tensor " << t->name << " has size of 0. Skipping it" << std::endl;
		return;
	}

	if (t->union_no < 0) {
		if (options.extern_init && t->initialize) {
			dst << "extern ";
		}
		else if (!options.only_init) {
			dst << "static ";
		}
	}

	dst << t->print_tensor_definition();
	if (t->initialize) {
		if (options.target_avr && t->isConst)
			dst << " PROGMEM";

		if (!options.extern_init) {
			dst << " = " << std::endl;
			t->print_tensor_initializer(dst);
		}
	}
	dst << ";" << std::endl;
}

void Graph::print_global_tensors(std::ostream& dst)
{
	// ununionized tensors
	LOG(TRACE) << "printing global tensors - ununionized " << std::endl;
	for (auto t : tensors) {
		LOG(TRACE) << "\t" << t->print_trace_dump() << std::endl;
		if (t->union_no < 0 && t->generate)
			this->print_tensor(t, dst);
	}

	LOG(TRACE) << "printing global tensors - unionized " << std::endl;
	for (unsigned u = 0; u < tensor_unions.size(); u++) {
		dst << "union tensor_union_" << u << " {" << std::endl;
		for (auto t : tensors) {
			if (t->union_no == static_cast<int32_t>(u))
				this->print_tensor(t, dst);
		}
		dst << "};" << std::endl;
		if (!no_globals) {
			dst << "static union tensor_union_" << u << " tu" << u << ";" << std::endl
			    << std::endl;
		}
	}
	LOG(TRACE) << "(done printing global tensors)" << std::endl;
}

void Graph::print_functions(std::ostream& dst)
{
	for (auto n : nodes) {
		// handle meta-nodes separately
		if (n->op_name == "graph_io")
			continue;
		dst << "/*" << std::endl;
		dst << " * Operand:           " << n->op_name << std::endl;
		dst << " * Name in ONNX file: " << n->onnx_name << std::endl;
		dst << " */" << std::endl;
		dst << "FUNC_PREFIX void ";
		dst << n->c_name() << "( ";
		n->print_function_parameters_definition(dst);
		dst << " )";
		dst << std::endl
		    << "{" << std::endl;

		n->print(dst);

		dst << "}" << std::endl
		    << std::endl;
	}
}

void Graph::print_includes(std::ostream& dst)
{
	dst << "#include <float.h>" << std::endl;
	dst << "#include <math.h>" << std::endl;
	dst << "#include <stdbool.h>" << std::endl;
	dst << "#include <stdint.h>" << std::endl;
	dst << "#include <string.h>" << std::endl;
	dst << "#include <stdlib.h>" << std::endl;
	dst << std::endl;

	dst << "#define MAX(X,Y) ( X > Y ? X : Y)" << std::endl;
	dst << "#define MIN(X,Y) ( X < Y ? X : Y)" << std::endl;
	dst << "#define CLIP(X,L) ( MAX(MIN(X,L), -L) )" << std::endl;
	dst << std::endl;

	// 'inline' functions are a C99 addition.
	dst << "#if __STDC_VERSION__ < 199901L" << std::endl;
	dst << "#define FUNC_PREFIX" << std::endl;
	dst << "#else" << std::endl;
	dst << "#define FUNC_PREFIX static inline" << std::endl;
	dst << "#endif" << std::endl;

	if (options.target_avr) {
		dst << "#include <avr/pgmspace.h>" << std::endl;
		dst << "#define RD_PROGMEM(x) pgm_read_byte(&(x));" << std::endl;
	}
}

void Graph::print_interface_function(std::ostream& dst, bool definition, const std::string& func_name)
{
	bool isfirst = true;
	// TODO: take the interface function name from the ONNX file name
	dst << "void " << func_name << "(";
	for (auto i : model.graph().input()) {
		/* TODO: FIXME: separate input tensors that are initialized
		 * or re-initializable (and therefore count as input), from
		 * the "actual" input data */
		Tensor* t = findTensor(i.name());

		if (t && t->isIO) {
			if (!isfirst)
				dst << ", ";
			else
				isfirst = false;

			// this makes scalars be printed as pointers
			// (since we don't differentiate between graph input and output
			// tensors, all scalars are passed as pointers.
			dst << t->print_tensor_as_const();
		}
	}

	// find the graph output node
	// loop through the output nodes' inputs, printing them
	Node* graph_out_node = findNodeByName("graph_output");
	if (graph_out_node == nullptr)
		ERROR("internal onnx2c error: no graph_output node");

	for (unsigned o = 0; o < graph_out_node->get_number_of_inputs(); o++) {
		Tensor* t = graph_out_node->get_input_tensor(o);

		if (t) {
			if (!isfirst)
				dst << ", ";
			else
				isfirst = false;

			// kludge... in contrived cases (like unit tests), the graph can have a constant vector as its ouput.
			// Since this is the last function we write anyway...
			t->isConst = false;
			dst << t->print_tensor();
		}
	}

	dst << ")";
	if (!definition) { // not definition, i.e. decalaration
		dst << ";" << std::endl;
		return;
	}

	// else: definition - print the rest
	dst << "{" << std::endl;

	// Print tensors here if no globals
	if (no_globals) {
		for (unsigned u = 0; u < tensor_unions.size(); u++) {
			INDT_1 << "union tensor_union_" << u << " tu" << u << ";" << std::endl;
		}
		dst << std::endl;
	}

	// since nodes were resolved from graph inputs in the order there were
	// node inputs resolved, the nodes vector is now sorted in order so that
	// we don't need to check dependancies :)
	for (auto n : nodes) {
		// handle meta-nodes separately
		if (n->op_name == "graph_io")
			continue;

		dst << "\t" << n->c_name() << "( ";
		n->print_function_parameters_callsite(dst);
		dst << ");" << std::endl;
	}

	dst << "}" << std::endl;
}
