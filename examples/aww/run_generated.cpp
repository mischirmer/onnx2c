#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" void entry(const float tensor_serving_default_input_1[1][1][49][10],
                      float tensor_StatefulPartitionedCall[1][12]);

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
		} else if (a == "--help" || a == "-h") {
			std::cout
			    << "usage: run_generated --bin-dir DIR --label-csv FILE [--limit N] [--verbose]\n"
			    << "                     [--input-scale S] [--input-zero-point Z]\n"
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

	const auto samples = load_mlperf_bin_manifest(bin_dir, label_csv, limit);
	if (samples.empty()) die("no samples loaded (check --bin-dir + --label-csv)");

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
	return 0;
}
