/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "solver/abstract_cache.h"

namespace DPF {

	//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
	//see also https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine
	//and https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
	struct BranchHashFunction {
		//todo check about overflows
		int operator()(Branch const& branch) const {
			int seed = int(branch.Depth());
			for (int i = 0; i < branch.Depth(); i++) {
				int code = branch[i];
				seed ^= code + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};

	//assumes that both inputs are in canonical representation
	struct BranchEquality {
		bool operator()(Branch const& branch1, Branch const& branch2) const {
			if (branch1.Depth() != branch2.Depth()) { return false; }
			for (int i = 0; i < branch1.Depth(); i++) {
				if (branch1[i] != branch2[i]) { return false; }
			}
			return true;
		}
	};

	//key: a branch
	//value: cached value contains the optimal value and the lower bound
	class BranchCache : public AbstractCache {
	public:
		BranchCache(int max_branch_length);

		//related to storing/retriving optimal branches
		bool IsOptimalAssignmentCached(BinaryData&, const Branch& branch, int depth, int num_nodes, int upper_bound);
		void StoreOptimalBranchAssignment(BinaryData&, const Branch& branch, std::shared_ptr<AssignmentContainer> optimal_solutions, int depth, int num_nodes, int upper_bound);
		std::shared_ptr<AssignmentContainer> RetrieveOptimalAssignment(BinaryData&, const Branch& branch, int depth, int num_nodes, int upper_bound);
		void TransferAssignmentsForEquivalentBranches(const BinaryData&, const Branch& branch_source, const BinaryData&, const Branch& branch_destination);//this updates branch_destination with all solutions from branch_source. Should only be done if the branches are equivalent.

		//related to storing/retrieving lower bounds
		void UpdateLowerBound(BinaryData&, const Branch& branch, int lower_bound, int depth, int num_nodes);
		int RetrieveLowerBound(BinaryData&, const Branch& branch, int depth, int num_nodes);

		//related to storing/retrieving discrimination budgets
		void UpdateDiscrimationBudget(const Branch& org_branch, Branch& this_branch, BinaryData& data, const Branch& branch, int depth, int num_nodes, int upper_bound);
		const DiscriminationBudget RetrieveBestBudgetBounds(BinaryData& data, const Branch& org_branch, const Branch& branch, int depth, int num_nodes, int upper_bound);

		//misc
		int NumEntries() const;

		void DisableLowerBounding();
		void DisableOptimalCaching();

	private:
		bool use_lower_bound_caching_;
		bool use_optimal_caching_;
		bool use_budget_caching_;

		std::vector<
			std::unordered_map<Branch, std::vector<CacheEntry>, BranchHashFunction, BranchEquality >
		> cache_; //cache_[i] is a hash table with branches of size i		
	};
}