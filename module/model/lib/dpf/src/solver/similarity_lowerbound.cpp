/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/

#include "solver/similarity_lowerbound.h"

namespace DPF {
	SimilarityLowerBoundComputer::SimilarityLowerBoundComputer(int max_depth, int size, int num_instances) :
		disabled_(!USE_SIM_BOUND) {
		Initialise(max_depth, size);
	}

	PairLowerBoundOptimal SimilarityLowerBoundComputer::ComputeLowerBound(BinaryData& data, Branch& branch, int depth, int size, int upper_bound, AbstractCache* cache) {
		if (disabled_) { return PairLowerBoundOptimal(0, false); }

		PairLowerBoundOptimal result(0, false);
		for (ArchiveEntry& entry : archive_[depth]) {
			int entry_lower_bound = cache->RetrieveLowerBound(entry.data, entry.branch, depth, size);
			if (entry.data.Size() > data.Size() && entry.data.Size() - data.Size() >= entry_lower_bound) { continue; } 

			DifferenceMetrics metrics = BinaryDataDifferenceComputer::ComputeDifferenceMetrics(entry.data, data);
			//result.lower_bound = std::max(result.lower_bound, entry_lower_bound - metrics.total_difference); // TODO reason about a sim. lower bound

			if (metrics.total_difference == 0) {
				cache->TransferAssignmentsForEquivalentBranches(entry.data, entry.branch, data, branch);
				if (cache->IsOptimalAssignmentCached(data, branch, depth, size, upper_bound)) {
					result.optimal = true;
					result.lower_bound = entry_lower_bound;
					break;
				}
			}
		}
		return result;
	}

	void SimilarityLowerBoundComputer::UpdateDiscrimationBudget(const Branch& org_branch, Branch& this_branch, BinaryData& data, const Branch& branch, const DataSummary& data_summary,
			int depth, int size, int upper_bound, AbstractCache* cache) {
		if (disabled_) { return; }
		DiscriminationBudget best_budget = this_branch.GetDiscriminationBudget();
		for (ArchiveEntry& entry : archive_[depth]) {
			// What if the solutions are computed for a different UB?
			const auto budget = cache->RetrieveBestBudgetBounds(entry.data, org_branch, entry.branch, depth, size, upper_bound);
			if (!budget.IsRestricted()) continue;
			DifferenceMetrics metrics = BinaryDataDifferenceComputer::ComputeDifferenceMetrics(entry.data, data);
			double min_balance = budget.min_balance - double(metrics.num_removals_group0) / data_summary.group0_size - double(metrics.num_additions_group1) / data_summary.group1_size;
			double max_balance = budget.max_balance + double(metrics.num_additions_group0) / data_summary.group0_size + double(metrics.num_removals_group1) / data_summary.group1_size;
			best_budget.Tighten(org_branch.GetDiscriminationBudget(), DiscriminationBudget(min_balance, max_balance));
			if (metrics.total_difference == 0) {
				//cache->TransferAssignmentsForEquivalentBranches(entry.data, entry.branch, data, branch);
			}
		}
		this_branch.SetDiscriminationBudget(best_budget);
	}

	void SimilarityLowerBoundComputer::UpdateArchive(BinaryData& data, Branch& branch, int depth) {
		if (disabled_) { return; }

		if (archive_[depth].size() < 2) { // TODO test with different values for this. What if larger values? More aggressive pruning might be useful
			archive_[depth].push_back(ArchiveEntry(data, branch));
		} else {
			GetMostSimilarStoredData(data, depth) = ArchiveEntry(data, branch);
		}
	}

	void SimilarityLowerBoundComputer::Initialise(int max_depth, int size) {
		if (disabled_) { return; }

		archive_.resize(max_depth + 1);
	}

	void SimilarityLowerBoundComputer::Disable() {
		disabled_ = true;
	}

	SimilarityLowerBoundComputer::ArchiveEntry& SimilarityLowerBoundComputer::GetMostSimilarStoredData(BinaryData& data, int depth) {
		runtime_assert(archive_[depth].size() > 0);

		SimilarityLowerBoundComputer::ArchiveEntry* best_entry = NULL;
		int best_similiarity_score = INT32_MAX;
		for (ArchiveEntry& archieve_entry : archive_[depth]) {
			int similiarity_score = BinaryDataDifferenceComputer::ComputeDifferenceMetrics(archieve_entry.data, data).total_difference;
			if (similiarity_score < best_similiarity_score) {
				best_entry = &archieve_entry;
				best_similiarity_score = similiarity_score;
			}
		}
		runtime_assert(best_similiarity_score < INT32_MAX);
		return *best_entry;
	}

}