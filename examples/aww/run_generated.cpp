#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" void entry(const float tensor_serving_default_input_1[1][1][49][10],
                      float tensor_StatefulPartitionedCall[1][12]);
extern "C" volatile bool TAMPERING_DETECTED;
extern "C" uint32_t TAMPERING_DETECTIONS;
extern "C" volatile bool FAULT_ENABLED;
extern "C" uint32_t FAULT_MODEL;
extern "C" uint32_t FAULT_LAYER_ID;
extern "C" uint32_t FAULT_INDEX;
extern "C" float FAULT_VALUE;
extern "C" uint32_t FAULT_N;
extern "C" uint32_t FAULT_STRIDE;
extern "C" volatile bool FAULT_INJECTED;
extern "C" uint32_t FAULT_INJECTIONS;
extern "C" const uint32_t SWEEP_LAYER_COUNT;
extern "C" const uint32_t SWEEP_LAYER_IDS[];
extern "C" const char* SWEEP_LAYER_OPS[];
extern "C" const uint64_t SWEEP_LAYER_OUT_ELEMS[];

namespace fs = std::filesystem;

static void die(const std::string& msg)
{
	std::cerr << "error: " << msg << "\n";
	std::exit(1);
}

static std::string trim(std::string s)
{
	auto is_space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
	while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
	while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) s.pop_back();
	return s;
}

static std::vector<std::string> split_csv_simple(const std::string& line)
{
	// Minimal CSV splitting (no quoted fields support; good enough for y_labels.csv)
	std::vector<std::string> out;
	std::stringstream ss(line);
	std::string tok;
	while (std::getline(ss, tok, ',')) out.push_back(trim(tok));
	return out;
}

struct SampleRef {
	fs::path path;
	int label;
};

struct RunStats {
	std::size_t samples = 0;
	std::size_t correct = 0;
	std::size_t correct_corrected = 0; // replace detected samples with baseline prediction
	std::size_t detected_samples = 0;
	bool tampering_detected = false;
	uint32_t tampering_detections = 0;
	bool fault_injected = false;
	uint32_t fault_injections = 0;
};

struct ProbeDiff {
	float max_abs = 0.0f;
	float mean_abs = 0.0f;
};

static std::optional<int> parse_int(const std::string& s)
{
	const std::string t = trim(s);
	if (t.empty()) return std::nullopt;
	std::size_t idx = 0;
	try {
		int v = std::stoi(t, &idx);
		if (idx != t.size()) return std::nullopt;
		return v;
	} catch (...) {
		return std::nullopt;
	}
}

static std::vector<SampleRef> load_mlperf_bin_manifest(const fs::path& bin_dir,
                                                       const fs::path& csv_path,
                                                       std::size_t limit)
{
	std::ifstream f(csv_path);
	if (!f) die("failed to open label csv: " + csv_path.string());

	std::vector<SampleRef> samples;
	samples.reserve(limit ? limit : 1024);

	std::string line;
	while (std::getline(f, line)) {
		if (limit && samples.size() >= limit) break;

		if (line.empty()) continue;
		if (line.rfind("#", 0) == 0) continue;

		const auto cols = split_csv_simple(line);
		if (cols.size() < 3) continue;
		const std::string fname = cols[0];
		const auto label_opt = parse_int(cols[2]);
		if (!label_opt) continue;

		const fs::path p = bin_dir / fname;
		if (!fs::exists(p)) continue;
		samples.push_back(SampleRef{p, *label_opt});
	}
	return samples;
}

static void read_sample_bin_int8_to_float(const fs::path& p, float x[1][1][49][10])
{
	constexpr std::size_t kElems = 49 * 10;
	constexpr std::size_t kBytes = kElems * sizeof(int8_t);

	std::ifstream f(p, std::ios::binary);
	if (!f) die("failed to open sample: " + p.string());

	int8_t buf[kElems];
	f.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(kBytes));
	if (!f) die("failed to read expected bytes from sample: " + p.string());

	for (int i = 0; i < 49; i++) {
		for (int j = 0; j < 10; j++) {
			x[0][0][i][j] = static_cast<float>(buf[i * 10 + j]);
		}
	}
}

static void dequantize_inplace(float x[1][1][49][10], float scale, int zero_point)
{
	for (int i = 0; i < 49; i++) {
		for (int j = 0; j < 10; j++) {
			x[0][0][i][j] = (x[0][0][i][j] - static_cast<float>(zero_point)) * scale;
		}
	}
}

static ProbeDiff probe_first_sample_diff(const std::vector<SampleRef>& samples,
                                         float input_scale,
                                         int input_zero_point)
{
	ProbeDiff d;
	if (samples.empty()) return d;

	float x[1][1][49][10];
	read_sample_bin_int8_to_float(samples[0].path, x);
	dequantize_inplace(x, input_scale, input_zero_point);

	float y0[1][12];
	float y1[1][12];

	const bool prev_fault_enabled = FAULT_ENABLED;

	FAULT_ENABLED = false;
	entry(x, y0);

	FAULT_ENABLED = prev_fault_enabled;
	entry(x, y1);

	float sum = 0.0f;
	for (int j = 0; j < 12; j++) {
		const float diff = std::fabs(y1[0][j] - y0[0][j]);
		d.max_abs = std::max(d.max_abs, diff);
		sum += diff;
	}
	d.mean_abs = sum / 12.0f;
	return d;
}

static int argmax12(const float y[1][12])
{
	int best_i = 0;
	float best_v = y[0][0];
	for (int i = 1; i < 12; i++) {
		if (y[0][i] > best_v) {
			best_v = y[0][i];
			best_i = i;
		}
	}
	return best_i;
}

static RunStats run_dataset(const std::vector<SampleRef>& samples, float input_scale, int input_zero_point)
{
	RunStats s;
	s.samples = samples.size();

	for (std::size_t i = 0; i < samples.size(); i++) {
		float x[1][1][49][10];
		float y_fault[1][12];
		float y_base[1][12];
		read_sample_bin_int8_to_float(samples[i].path, x);
		dequantize_inplace(x, input_scale, input_zero_point);

		// Run with current fault/ABFT configuration, but measure detection per-sample
		// by resetting the global counter before each inference.
		TAMPERING_DETECTED = false;
		TAMPERING_DETECTIONS = 0;
		FAULT_INJECTED = false;
		FAULT_INJECTIONS = 0;
		entry(x, y_fault);

		const bool detected = TAMPERING_DETECTED || (TAMPERING_DETECTIONS > 0);
		if (detected) s.detected_samples++;

		const int pred_fault = argmax12(y_fault);
		if (pred_fault == samples[i].label) s.correct++;

		// "Corrected" accuracy: if ABFT flags this sample, fall back to a baseline
		// prediction (same model, but with FAULT_ENABLED=false).
		if (detected) {
			const bool prev_fault_enabled = FAULT_ENABLED;
			FAULT_ENABLED = false;
			entry(x, y_base);
			FAULT_ENABLED = prev_fault_enabled;
			const int pred_base = argmax12(y_base);
			if (pred_base == samples[i].label) s.correct_corrected++;
		} else {
			if (pred_fault == samples[i].label) s.correct_corrected++;
		}
	}

	s.tampering_detected = TAMPERING_DETECTED;
	s.tampering_detections = TAMPERING_DETECTIONS;
	s.fault_injected = FAULT_INJECTED;
	s.fault_injections = FAULT_INJECTIONS;
	return s;
}

static std::vector<float> parse_float_list(const std::string& csv)
{
	std::vector<float> out;
	std::stringstream ss(csv);
	std::string tok;
	while (std::getline(ss, tok, ',')) {
		tok = trim(tok);
		if (tok.empty()) continue;
		out.push_back(std::stof(tok));
	}
	return out;
}

static std::vector<std::string> parse_string_list(const std::string& csv)
{
	std::vector<std::string> out;
	std::stringstream ss(csv);
	std::string tok;
	while (std::getline(ss, tok, ',')) {
		tok = trim(tok);
		if (tok.empty()) continue;
		out.push_back(tok);
	}
	return out;
}

static uint32_t fault_model_from_name(const std::string& name)
{
	if (name == "single_point") return 0;
	if (name == "trivial") return 1;
	if (name == "checkered") return 2;
	die("unknown fault model: " + name);
	return 0;
}

int main(int argc, char** argv)
{
	fs::path bin_dir;
	fs::path label_csv;
	std::size_t limit = 0;
	bool verbose = false;
	float input_scale =
#ifdef INPUT_SCALE
	    static_cast<float>(INPUT_SCALE)
#else
	    1.0f
#endif
	    ;
	int input_zero_point =
#ifdef INPUT_ZERO_POINT
	    static_cast<int>(INPUT_ZERO_POINT)
#else
	    0
#endif
	    ;

	bool fault_enable_seen = false;
	bool fault_param_seen = false;

	bool sweep = false;
	std::size_t sweep_indexes = 10;
	std::string sweep_patterns_csv = "single_point,trivial,checkered";
	std::string sweep_values_csv = "5, 10, 1e3";
	bool sweep_progress = true;
	uint32_t sweep_seed = 12345;
	bool sweep_debug = false;

	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		if ((a == "--bin-dir" || a == "-b") && i + 1 < argc) {
			bin_dir = argv[++i];
		} else if ((a == "--label-csv" || a == "-y") && i + 1 < argc) {
			label_csv = argv[++i];
		} else if ((a == "--limit" || a == "-n") && i + 1 < argc) {
			limit = static_cast<std::size_t>(std::stoul(argv[++i]));
		} else if (a == "--input-scale" && i + 1 < argc) {
			input_scale = std::stof(argv[++i]);
		} else if (a == "--input-zero-point" && i + 1 < argc) {
			input_zero_point = std::stoi(argv[++i]);
		} else if (a == "--verbose" || a == "-v") {
			verbose = true;
		} else if (a == "--fault-enable") {
			FAULT_ENABLED = true;
			fault_enable_seen = true;
		} else if (a == "--fault-layer" && i + 1 < argc) {
			FAULT_LAYER_ID = static_cast<uint32_t>(std::stoul(argv[++i]));
			fault_param_seen = true;
		} else if (a == "--fault-layer-any") {
			FAULT_LAYER_ID = 0xFFFFFFFFu;
			fault_param_seen = true;
		} else if (a == "--fault-index" && i + 1 < argc) {
			FAULT_INDEX = static_cast<uint32_t>(std::stoul(argv[++i]));
			fault_param_seen = true;
		} else if (a == "--fault-value" && i + 1 < argc) {
			FAULT_VALUE = std::stof(argv[++i]);
			fault_param_seen = true;
		} else if (a == "--fault-n" && i + 1 < argc) {
			FAULT_N = static_cast<uint32_t>(std::stoul(argv[++i]));
			fault_param_seen = true;
		} else if (a == "--fault-stride" && i + 1 < argc) {
			FAULT_STRIDE = static_cast<uint32_t>(std::stoul(argv[++i]));
			fault_param_seen = true;
		} else if (a == "--fault-model" && i + 1 < argc) {
			const std::string m = argv[++i];
			if (m == "single_point")
				FAULT_MODEL = 0;
			else if (m == "trivial")
				FAULT_MODEL = 1;
			else if (m == "checkered")
				FAULT_MODEL = 2;
			else
				die("unknown fault model: " + m);
			fault_param_seen = true;
		} else if (a == "--sweep") {
			sweep = true;
		} else if (a == "--sweep-indexes" && i + 1 < argc) {
			sweep_indexes = static_cast<std::size_t>(std::stoul(argv[++i]));
		} else if (a == "--sweep-patterns" && i + 1 < argc) {
			sweep_patterns_csv = argv[++i];
		} else if (a == "--sweep-values" && i + 1 < argc) {
			sweep_values_csv = argv[++i];
		} else if (a == "--sweep-seed" && i + 1 < argc) {
			sweep_seed = static_cast<uint32_t>(std::stoul(argv[++i]));
		} else if (a == "--sweep-debug") {
			sweep_debug = true;
		} else if (a == "--no-progress") {
			sweep_progress = false;
		} else if (a == "--help" || a == "-h") {
			std::cout
			    << "usage: run_generated --bin-dir DIR --label-csv FILE [--limit N] [--verbose]\n"
			    << "                     [--input-scale S] [--input-zero-point Z]\n"
			    << "                     [--fault-enable --fault-model {single_point,trivial,checkered} (--fault-layer L|--fault-layer-any) --fault-index I --fault-value V]\n"
			    << "                     [--fault-n N] [--fault-stride S]\n"
			    << "                     [--sweep --sweep-indexes N --sweep-patterns CSV --sweep-values CSV --sweep-seed SEED] [--sweep-debug] [--no-progress]\n"
			    << "\n"
			    << "Expects MLPerf Tiny KWS bins: each .bin is 49x10 int8 MFCC (490 bytes).\n"
			    << "The label CSV is the MLPerf y_labels.csv with lines like:\n"
			    << "  tst_000000_Stop_7.bin, 12, 7\n";
			return 0;
		} else {
			die("unknown arg: " + a);
		}
	}

	if (bin_dir.empty()) die("missing --bin-dir");
	if (label_csv.empty()) die("missing --label-csv");

	TAMPERING_DETECTED = false;
	TAMPERING_DETECTIONS = 0;
	FAULT_INJECTED = false;
	FAULT_INJECTIONS = 0;

	// Convenience: if a fault configuration is provided, enable it even if
	// --fault-enable was forgotten.
	if (!fault_enable_seen && fault_param_seen) {
		FAULT_ENABLED = true;
	}

	const auto samples = load_mlperf_bin_manifest(bin_dir, label_csv, limit);
	if (samples.empty()) die("no samples loaded (check --bin-dir + --label-csv)");

	if (sweep) {
		const auto patterns = parse_string_list(sweep_patterns_csv);
		const auto values = parse_float_list(sweep_values_csv);
		if (patterns.empty()) die("--sweep-patterns has no entries");
		if (values.empty()) die("--sweep-values has no entries");
		if (SWEEP_LAYER_COUNT == 0) die("SWEEP_LAYER_COUNT == 0 (no sweepable layers)");

		const std::size_t total_runs =
		    (std::size_t)SWEEP_LAYER_COUNT * patterns.size() * values.size() * sweep_indexes;
		std::size_t done_runs = 0;
		std::mt19937 rng(sweep_seed);

		// Baseline (fault-free) accuracy for this binary/config (useful for plotting).
		{
			const bool prev_fault_enabled = FAULT_ENABLED;
			FAULT_ENABLED = false;
			const auto st0 = run_dataset(samples, input_scale, input_zero_point);
			FAULT_ENABLED = prev_fault_enabled;
			const double acc0 = st0.samples ? (double)st0.correct / (double)st0.samples : 0.0;
			const double acc0_corr =
			    st0.samples ? (double)st0.correct_corrected / (double)st0.samples : 0.0;
			std::cout << "baseline_accuracy=" << acc0 << " baseline_accuracy_corrected=" << acc0_corr
			          << "\n";
		}

		std::cout << "sweep_layers=" << SWEEP_LAYER_COUNT << " sweep_patterns=" << patterns.size()
		          << " sweep_indexes=" << sweep_indexes << " sweep_values=" << values.size() << "\n";

		for (uint32_t li = 0; li < SWEEP_LAYER_COUNT; li++) {
			const uint32_t layer_id = SWEEP_LAYER_IDS[li];
			const std::string layer_op = SWEEP_LAYER_OPS[li] ? SWEEP_LAYER_OPS[li] : "";
			const uint64_t out_elems = SWEEP_LAYER_OUT_ELEMS[li];
			if (out_elems == 0) die("SWEEP_LAYER_OUT_ELEMS == 0 for layer " + std::to_string(layer_id));
			std::uniform_int_distribution<uint64_t> dist(0ull, out_elems - 1ull);
			for (const auto& pat : patterns) {
				const uint32_t model = fault_model_from_name(pat);
				for (float val : values) {
					double acc_min = 1.0, acc_max = 0.0, acc_sum = 0.0;
					double acc_corr_min = 1.0, acc_corr_max = 0.0, acc_corr_sum = 0.0;
					double det_min = 1e30, det_max = 0.0, det_sum = 0.0;
					uint32_t inj_min = UINT32_MAX, inj_max = 0;
					uint64_t faults_with_injection = 0;
					float probe_max_abs_max = 0.0f;
					float probe_mean_abs_max = 0.0f;

					for (std::size_t t = 0; t < sweep_indexes; t++) {
						done_runs++;
						if (sweep_progress) {
							const int pct =
							    (total_runs == 0) ? 100 : (int)((done_runs * 100) / total_runs);
							std::cerr << "\rprogress " << pct << "% (" << done_runs << "/"
							          << total_runs << ")" << std::flush;
						}

						// Configure fault
						FAULT_ENABLED = true;
						FAULT_MODEL = model;
						FAULT_LAYER_ID = layer_id;
						FAULT_VALUE = val;
						FAULT_INDEX = static_cast<uint32_t>(dist(rng));

						// Reset per-run counters
						TAMPERING_DETECTED = false;
						TAMPERING_DETECTIONS = 0;
						FAULT_INJECTED = false;
						FAULT_INJECTIONS = 0;

						const ProbeDiff probe = probe_first_sample_diff(samples, input_scale, input_zero_point);
						probe_max_abs_max = std::max(probe_max_abs_max, probe.max_abs);
						probe_mean_abs_max = std::max(probe_mean_abs_max, probe.mean_abs);

						const auto st = run_dataset(samples, input_scale, input_zero_point);
						const double acc = st.samples ? (double)st.correct / (double)st.samples : 0.0;
						const double acc_corr =
						    st.samples ? (double)st.correct_corrected / (double)st.samples : 0.0;
						const double det = (double)st.tampering_detections;
						inj_min = std::min(inj_min, st.fault_injections);
						inj_max = std::max(inj_max, st.fault_injections);
						if (st.fault_injections > 0) faults_with_injection++;
						acc_min = std::min(acc_min, acc);
						acc_max = std::max(acc_max, acc);
						acc_sum += acc;
						acc_corr_min = std::min(acc_corr_min, acc_corr);
						acc_corr_max = std::max(acc_corr_max, acc_corr);
						acc_corr_sum += acc_corr;
						det_min = std::min(det_min, det);
						det_max = std::max(det_max, det);
						det_sum += det;
					}

					const double acc_mean = acc_sum / (double)sweep_indexes;
					const double acc_corr_mean = acc_corr_sum / (double)sweep_indexes;
					const double det_mean = det_sum / (double)sweep_indexes;

					std::cout << "layer=" << layer_id << " op=" << layer_op << " pattern=" << pat
					          << " value=" << val
					          << " idx_trials=" << sweep_indexes << " acc_min=" << acc_min
					          << " acc_max=" << acc_max << " acc_mean=" << acc_mean
					          << " acc_corr_min=" << acc_corr_min
					          << " acc_corr_max=" << acc_corr_max
					          << " acc_corr_mean=" << acc_corr_mean
					          << " det_min=" << det_min << " det_max=" << det_max
					          << " det_mean=" << det_mean
					          << " inj_min=" << inj_min << " inj_max=" << inj_max
					          << " inj_trials=" << faults_with_injection
					          << " probe_max_abs_max=" << probe_max_abs_max
					          << " probe_mean_abs_max=" << probe_mean_abs_max << "\n";

					if (sweep_debug) {
						std::cerr << "debug layer=" << layer_id << " op=" << layer_op
						          << " out_elems=" << out_elems
						          << " inj_trials=" << faults_with_injection << "/" << sweep_indexes
						          << " inj_min=" << inj_min << " inj_max=" << inj_max << "\n";
					}
				}
			}
		}
		if (sweep_progress) std::cerr << "\rprogress 100% (" << done_runs << "/" << total_runs << ")\n";
		return 0;
	}

	std::size_t correct = 0;
	for (std::size_t i = 0; i < samples.size(); i++) {
		float x[1][1][49][10];
		float y[1][12];
		read_sample_bin_int8_to_float(samples[i].path, x);
		dequantize_inplace(x, input_scale, input_zero_point);
		entry(x, y);

		const int pred = argmax12(y);
		const int gold = samples[i].label;
		if (pred == gold) correct++;

		if (verbose) {
			std::cout << i << "," << samples[i].path.filename().string() << ",pred=" << pred
			          << ",gold=" << gold << "\n";
		}
	}

	const double acc = static_cast<double>(correct) / static_cast<double>(samples.size());
	std::cout << "samples=" << samples.size() << " correct=" << correct << " accuracy=" << acc
	          << "\n";
	// Note: "corrected" accuracy is only meaningful when ABFT is enabled in codegen.
	// We compute it by replacing detected samples with baseline (fault-free) predictions.
	{
		// Re-run with the same CLI-configured fault settings to get corrected stats.
		const auto st = run_dataset(samples, input_scale, input_zero_point);
		const double acc_corr =
		    st.samples ? (double)st.correct_corrected / (double)st.samples : 0.0;
		std::cout << "accuracy_corrected=" << acc_corr << " detected_samples=" << st.detected_samples
		          << "\n";
	}
	std::cout << "tampering_detected=" << (TAMPERING_DETECTED ? "true" : "false")
	          << " tampering_detections=" << TAMPERING_DETECTIONS << "\n";
	std::cout << "fault_injected=" << (FAULT_INJECTED ? "true" : "false")
	          << " fault_injections=" << FAULT_INJECTIONS << "\n";
	return 0;
}
