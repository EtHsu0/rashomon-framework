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
	struct DatasetHashFunction {
		//todo check about overflows
		long long operator()(const BinaryData& data) const {
			return data.GetHash();
		}

		static long long ComputeHashForData(const BinaryData& data) {
			long long seed = int(data.Size());
			for(int i = 0; i < data.Size(); i++) {
				long long code = data.GetInstance(i)->GetID();
				seed ^= code + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};

	//assumes that both inputs are in canonical representation
	struct DatasetEquality {
		bool operator()(const BinaryData& data1, const BinaryData& data2) const {
			runtime_assert(data1.NumLabels() == data2.NumLabels() && data1.NumFeatures() == data2.NumFeatures());

			if (data1.GetHash() != data2.GetHash()) { return false; }

			//basic check on the size
			if (data1.Size() != data2.Size()) { return false; }
			//basic check on the size of each individual label
			for (int label = 0; label < data1.NumLabels(); label++) {
				if(data1.NumInstancesForLabel(label) != data2.NumInstancesForLabel(label)) return false;
			}
			for (int group = 0; group < data1.NumGroups(); group++) {
				if (data1.NumInstancesForGroup(group) != data2.NumInstancesForGroup(group)) { return false; }
			}

			//now compare individual feature vectors
			//note that the indicies are kept sorted in the data
			for (int i = 0; i < data1.Size(); i++) {
				int code1 = data1.GetInstance(i)->GetID();
				int code2 = data2.GetInstance(i)->GetID();
				if (code1 != code2) { return false; }
			}
			return true;
		}
	};

	//key: a dataset
	//value: cached value contains the optimal value and the lower bound
	class DatasetCache : public AbstractCache {
	public:
		DatasetCache(int max_branch_length);

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

		//private:
		bool use_lower_bound_caching_;
		bool use_optimal_caching_;

		std::vector<
			std::unordered_map<BinaryData, std::vector<CacheEntry>, DatasetHashFunction, DatasetEquality >
		> cache_; //cache_[i] is a hash table with datasets of size i	

		//we store a few iterators that were previous used in case they will be used in the future
		//useful when we query the cache multiple times for the exact same query
		struct PairIteratorBranch {
			std::unordered_map<BinaryData, std::vector<CacheEntry>, DatasetHashFunction, DatasetEquality >::iterator iter;
			Branch branch;
		};

		void InvalidateStoredIterators(BinaryData& data);
		std::unordered_map<BinaryData, std::vector<CacheEntry>, DatasetHashFunction, DatasetEquality >::iterator FindIterator(BinaryData& data, const Branch& branch);
		std::vector<std::deque<PairIteratorBranch> > stored_iterators_;
	};

}