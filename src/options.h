/* This file is part of onnx2c.
 *
 * Onnx2c code gereration options.
 */

#pragma once

#include <map>
#include <stdint.h>
#include <string>
#include <vector>

enum class im2col_mode {
	HEURISTIC,
	ALL,
};

struct onnx2c_opts {
	bool target_avr = false;
	bool no_globals = false;
	bool extern_init = false;
	bool only_init = false;
	bool opt_unionize = true;
	bool opt_fold_casts = true;
	bool opt_im2col = false;
	im2col_mode opt_im2col_mode = im2col_mode::HEURISTIC;

	// ABFT options (from abft branch)
	bool conv_im2col = false; // Emit Conv as im2col + dot-product (matmul-style)
	bool abft_gemm = false;   // Add ABFT checks around gemm-like dot-products
	bool abyzft_gemm = false; // AByzFT: randomized scaling around gemm-like dot-products
	bool freivalds_gemm = false; // Freivalds check (random {0,1} vector) around gemm-like dot-products
	uint32_t freivalds_checks = 1; // Number of independent Freivalds checks (default: 1)
	bool gvfa_gemm = false; // GVFA: Freivalds variant with Gaussian random vectors
	uint32_t gvfa_checks = 1; // Number of independent GVFA checks (default: 1)
	uint32_t abft_mtile = 64; // Output-channel tile size for ABFT/Freivalds/AByzFT
	float abft_eps = 1e-3f;   // Relative tolerance for checksum verification
	bool abft_weight_checksums_compiletime = true; // Precompute ABFT weight checksums (default)
/*
 * logging levels are
 * cmd line     aixlog     Use
 * -------------------------------
 *   -          FATAL
 *   0          ERROR      Bad input or missing feature in onnx2c.
 *   1          WARNING    Valid input, but onnx2c output might not be conformant.
 *   2          INFO       Generic info, warnigns about suboptimal input.
 *   3          DEBUG      Notes on generated nodes and tensors.
 *   4          TRACE      Detailed info on generated nodes and tensors.
 */
#ifndef DEFAULT_LOG_LEVEL
#define DEFAULT_LOG_LEVEL 2
#endif
	int logging_level = DEFAULT_LOG_LEVEL; // Default level set by CMake. 1 in release, 4 in debug builds
	std::string input_file;
	std::string interface_func_name = "entry";
	std::map<std::string, uint32_t> dim_defines;

	// Save the raw command line arguments such that they can be printed
	// into the generated source file.
	std::vector<std::string> command_line_args;
};

extern struct onnx2c_opts options;

/* Parse command line, fill the global 'options' struct
 * with the results
 */
void parse_cmdline_options(int argc, const char* argv[]);
