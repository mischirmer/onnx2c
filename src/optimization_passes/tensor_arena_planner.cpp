#include "tensor_arena_planner.h"

#include "error.h"
#include "tensor.h"

#include <algorithm>
#include <map>
#include <set>

namespace toC {
namespace {

size_t checked_add(size_t a, size_t b, const char* context)
{
	if (a > SIZE_MAX - b)
		ERROR(context);
	return a + b;
}

} // namespace

size_t align_up(size_t value, size_t alignment)
{
	if (alignment == 0)
		ERROR("align_up called with zero alignment");
	if ((alignment & (alignment - 1)) != 0)
		ERROR("align_up currently requires power-of-two alignment");
	size_t remainder = value % alignment;
	if (remainder == 0)
		return value;
	return checked_add(value, alignment - remainder, "align_up overflow");
}

bool address_ranges_overlap(size_t offset_a, size_t size_a, size_t offset_b, size_t size_b)
{
	size_t end_a = checked_add(offset_a, size_a, "address range overflow");
	size_t end_b = checked_add(offset_b, size_b, "address range overflow");
	return offset_a < end_b && offset_b < end_a;
}

bool allocations_conflict(const ArenaAllocation& a, const ArenaAllocation& b)
{
	TensorLifetime life_a{a.tensor, a.first_use, a.last_use, a.size, a.alignment};
	TensorLifetime life_b{b.tensor, b.first_use, b.last_use, b.size, b.alignment};
	return lifetimes_overlap(life_a, life_b) && address_ranges_overlap(a.offset, a.size, b.offset, b.size);
}

bool validate_arena_plan(const ArenaPlan& plan, const std::vector<TensorLifetime>& lifetimes, std::string* reason)
{
	std::map<const Tensor*, TensorLifetime> lifetime_by_tensor;
	for (const TensorLifetime& lifetime : lifetimes)
		lifetime_by_tensor[lifetime.tensor] = lifetime;

	std::set<const Tensor*> seen;
	for (const ArenaAllocation& allocation : plan.allocations) {
		if (allocation.tensor == nullptr) {
			if (reason)
				*reason = "allocation without tensor";
			return false;
		}
		if (!seen.insert(allocation.tensor).second) {
			if (reason)
				*reason = "duplicate tensor allocation";
			return false;
		}
		auto life_it = lifetime_by_tensor.find(allocation.tensor);
		if (life_it == lifetime_by_tensor.end()) {
			if (reason)
				*reason = "allocation for unknown tensor";
			return false;
		}
		if (allocation.alignment == 0 || (allocation.alignment & (allocation.alignment - 1)) != 0) {
			if (reason)
				*reason = "invalid allocation alignment";
			return false;
		}
		if ((allocation.offset % allocation.alignment) != 0) {
			if (reason)
				*reason = "misaligned allocation offset";
			return false;
		}
		if (allocation.size != life_it->second.size_bytes || allocation.alignment != life_it->second.alignment || allocation.first_use != life_it->second.first_use || allocation.last_use != life_it->second.last_use) {
			if (reason)
				*reason = "allocation metadata does not match tensor lifetime";
			return false;
		}
		if (allocation.offset > plan.arena_size) {
			if (reason)
				*reason = "allocation offset beyond arena size";
			return false;
		}
		if (allocation.size > 0) {
			if (allocation.offset > SIZE_MAX - allocation.size) {
				if (reason)
					*reason = "allocation end overflow";
				return false;
			}
			if (allocation.offset + allocation.size > plan.arena_size) {
				if (reason)
					*reason = "allocation exceeds arena size";
				return false;
			}
		}
	}

	if (seen.size() != lifetimes.size()) {
		if (reason)
			*reason = "eligible tensor count does not match allocation count";
		return false;
	}

	for (size_t i = 0; i < plan.allocations.size(); i++) {
		for (size_t j = i + 1; j < plan.allocations.size(); j++) {
			if (allocations_conflict(plan.allocations[i], plan.allocations[j])) {
				if (reason)
					*reason = "simultaneously-live tensors overlap in memory";
				return false;
			}
		}
	}

	return true;
}

ArenaPlan TensorArenaPlanner::plan(const std::vector<TensorLifetime>& lifetimes) const
{
	std::vector<TensorLifetime> sorted = lifetimes;
	std::sort(sorted.begin(), sorted.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
		if (a.size_bytes != b.size_bytes)
			return a.size_bytes > b.size_bytes;
		if (a.first_use != b.first_use)
			return a.first_use < b.first_use;
		if (a.last_use != b.last_use)
			return a.last_use < b.last_use;
		return a.tensor->name < b.tensor->name;
	});

	ArenaPlan plan;
	for (const TensorLifetime& lifetime : sorted) {
		ArenaAllocation allocation;
		allocation.tensor = lifetime.tensor;
		allocation.size = lifetime.size_bytes;
		allocation.alignment = lifetime.alignment;
		allocation.first_use = lifetime.first_use;
		allocation.last_use = lifetime.last_use;

		std::vector<size_t> candidates{0};
		for (const ArenaAllocation& placed : plan.allocations) {
			TensorLifetime placed_lifetime{placed.tensor, placed.first_use, placed.last_use, placed.size, placed.alignment};
			if (!lifetimes_overlap(lifetime, placed_lifetime))
				continue;
			candidates.push_back(checked_add(placed.offset, placed.size, "candidate offset overflow"));
		}
		std::sort(candidates.begin(), candidates.end());
		candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

		bool placed = false;
		for (size_t candidate : candidates) {
			candidate = align_up(candidate, allocation.alignment);
			allocation.offset = candidate;
			bool conflict = false;
			for (const ArenaAllocation& other : plan.allocations) {
				if (allocations_conflict(allocation, other)) {
					conflict = true;
					break;
				}
			}
			if (conflict)
				continue;

			plan.allocations.push_back(allocation);
			plan.arena_size = std::max(plan.arena_size, checked_add(allocation.offset, allocation.size, "arena size overflow"));
			placed = true;
			break;
		}

		if (!placed)
			ERROR("failed to place tensor in arena plan");
	}

	return plan;
}

} // namespace toC
