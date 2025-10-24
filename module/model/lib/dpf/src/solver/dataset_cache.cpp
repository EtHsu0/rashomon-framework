/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/dataset_cache.h"

namespace DPF {

	DatasetCache::DatasetCache(int num_instances) :
		cache_(num_instances + 1),
		use_lower_bound_caching_(false),
		use_optimal_caching_(true),
		stored_iterators_(num_instances + 1) {
	}

	bool DatasetCache::IsOptimalAssignmentCached(BinaryData& data, const Branch& branch, int depth, int num_nodes, int upper_bound) {
		runtime_assert(depth <= num_nodes);

		if (!use_optimal_caching_) { return false; }

		auto& hashmap = cache_[data.Size()];
		auto iter = FindIterator(data, branch);

		if (iter == hashmap.end()) { return false; }

		for (CacheEntry& entry : iter->second) {
			if (entry.GetNodeBudget() == num_nodes && entry.GetDepthBudget() == depth 
				&& entry.GetDiscriminationBudget() == branch.GetDiscriminationBudget()
				&& entry.GetUpperBound() >= upper_bound) { 
				return entry.IsOptimal();
			}
		}
		return false;
	}

	void DatasetCache::StoreOptimalBranchAssignment(BinaryData& data, const Branch& b, std::shared_ptr<AssignmentContainer> optimal_solutions_c, int depth, int num_nodes, int upper_bound) {
		runtime_assert(depth <= num_nodes && num_nodes > 0);

		if (!use_optimal_caching_) { return; }

		auto& hashmap = cache_[data.Size()];
		auto iter_vector_entry = FindIterator(data, b);// hashmap.find(data);

		int optimal_node_depth = std::min(depth, optimal_solutions_c->NumNodes()); //this is an estimate of the depth, it could be lower actually. We do not consider lower for simplicity, but it would be good to consider it as well.

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry> vector_entry;
			for (int node_budget = optimal_solutions_c->NumNodes(); node_budget <= num_nodes; node_budget++) {
				for (int depth_budget = optimal_node_depth; depth_budget <= std::min(depth, node_budget); depth_budget++) {
					CacheEntry entry(depth_budget, node_budget, upper_bound, b.GetDiscriminationBudget(), optimal_solutions_c, optimal_solutions_c->GetBestBounds());
					vector_entry.push_back(entry);
				}
			}
			if (!data.IsHashSet()) { data.SetHash(DatasetHashFunction::ComputeHashForData(data)); }
			cache_[data.Size()].insert(std::pair<BinaryData, std::vector<CacheEntry> >(data, vector_entry));
			InvalidateStoredIterators(data);
		} else {
			//this sol is valid for size=[opt.NumNodes, num_nodes] and depths d=min(size, depth)

			//now we need to see if other node budgets have been seen before. 
			//For each budget that has been seen, update it;
			std::vector<std::vector<bool> > budget_seen(size_t(num_nodes) + 1, std::vector<bool>(depth + 1, false));
			for (CacheEntry& entry : iter_vector_entry->second) {
				//todo enable this here! //runtime_assert(optimal_node.Misclassifications() >= entry.GetLowerBound() || optimal_node.NumNodes() > entry.GetNodeBudget());

				//I believe it rarely happens that we receive a solution with less nodes than 'num_nodes', but it is possible
				if (optimal_solutions_c->NumNodes() <= entry.GetNodeBudget() && entry.GetNodeBudget() <= num_nodes
					&& optimal_node_depth <= entry.GetDepthBudget() && entry.GetDepthBudget() <= depth) {
					/*if (!(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_solutions->Misclassifications())) {
						std::cout << "opt node: " << optimal_solutions->NumNodes() << ", " << optimal_node.misclassification_score << "\n";
						std::cout << "\tnum nodes: " << num_nodes << "\n";
						std::cout << entry.GetNodeBudget() << ", " << entry.GetOptimalValue() << "\n";
					}*/ // TODO fix this
					// runtime_assert(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_node.Misclassifications()); TODO enable this

					if (entry.GetDiscriminationBudget() == b.GetDiscriminationBudget()
						&& entry.GetUpperBound() == upper_bound) {
						budget_seen[entry.GetNodeBudget()][entry.GetDepthBudget()] = true;
						if (!entry.IsOptimal()) {
							entry.SetOptimalSolutions(optimal_solutions_c);
						}
					}

					runtime_assert(entry.GetDepthBudget() <= entry.GetNodeBudget()); //fix the case when it turns out that more nodes do not give a better result...e.g., depth 4 and num nodes 4, but a solution with three nodes found...that solution is then optimal for depth 3 as well...need to update but lazy now
				}
			}
			//create entries for those which were not seen
			//note that most of the time this loop only does one iteration since usually using the full node budget gives better results
			for (int node_budget = optimal_solutions_c->NumNodes(); node_budget <= num_nodes; node_budget++) {
				for (int depth_budget = optimal_node_depth; depth_budget <= std::min(depth, node_budget); depth_budget++) {
					if (!budget_seen[node_budget][depth_budget]) {
						CacheEntry entry(depth_budget, node_budget, upper_bound, b.GetDiscriminationBudget(), optimal_solutions_c, optimal_solutions_c->GetBestBounds());
						iter_vector_entry->second.push_back(entry);
						runtime_assert(entry.GetDepthBudget() <= entry.GetNodeBudget()); //todo no need for this assert
					}
				}
			}
		}
		//TODO: the cache needs to invalidate out solutions that are dominated, i.e., with the same objective value but less nodes
		//or I need to rethink this caching to include exactly num_nodes -> it might be strange that we ask for five nodes and get UNSAT, while with four nodes it gives a solution
		//I am guessing that the cache must store exactly num_nodes, and then outside the loop when we find that the best sol has less node, we need to insert that in the cache?
		//and mark all solutions with more nodes as infeasible, i.e., some high cost
		
		// TODO check solution quality of cache with new found solution
		//auto opt = RetrieveOptimalAssignment(data, b, depth, num_nodes);
		//runtime_assert(opt == optimal_solutions);
	}

	void DatasetCache::TransferAssignmentsForEquivalentBranches(const BinaryData&, const Branch& branch_source, const BinaryData&, const Branch& branch_destination) {
		return; //no need to transfer when caching datasets
	}

	std::shared_ptr<AssignmentContainer> DatasetCache::RetrieveOptimalAssignment(BinaryData& data, const Branch& b, int depth, int num_nodes, int upper_bound) {
		auto& hashmap = cache_[data.Size()];
		auto iter = FindIterator(data, b);// hashmap.find(data);

		if (iter == hashmap.end()) { return std::shared_ptr<AssignmentContainer>(nullptr); }

		for (CacheEntry& entry : iter->second) {
			if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes 
				&& entry.GetDiscriminationBudget() == b.GetDiscriminationBudget() 
				&& entry.GetUpperBound() >= upper_bound && entry.IsOptimal() ) {
				return entry.GetOptimalSolution();
			}
		}
		return std::shared_ptr<AssignmentContainer>(nullptr);
	}

	void DatasetCache::UpdateLowerBound(BinaryData& data, const Branch& branch, int lower_bound, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		if (!use_lower_bound_caching_) { return; }

		auto& hashmap = cache_[data.Size()];
		auto iter_vector_entry = FindIterator(data, branch);// hashmap.find(data);

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry> vector_entry(1, CacheEntry(depth, num_nodes, INT32_MAX)); // TODO decide what discrimination values to set here
			vector_entry[0].UpdateLowerBound(lower_bound);
			if (!data.IsHashSet()) { data.SetHash(DatasetHashFunction::ComputeHashForData(data)); }
			cache_[data.Size()].insert(std::pair<BinaryData, std::vector<CacheEntry> >(data, vector_entry));
			InvalidateStoredIterators(data);
		} else {
			//now we need to see if this node node_budget has been seen before. 
			//If it was seen, update it; otherwise create a new entry
			bool found_corresponding_entry = false;
			for (CacheEntry& entry : iter_vector_entry->second) {
				if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes
					&& entry.GetDiscriminationBudget() == branch.GetDiscriminationBudget()) { // TODO reason about non-equal budgets
					entry.UpdateLowerBound(lower_bound);
					found_corresponding_entry = true;
					break;
				}
			}

			if (!found_corresponding_entry) {
				CacheEntry entry(depth, num_nodes, INT32_MAX); // TODO decide what discrimination values to set here
				entry.UpdateLowerBound(lower_bound);
				iter_vector_entry->second.push_back(entry);
			}
		}
	}

	int DatasetCache::RetrieveLowerBound(BinaryData& data, const Branch& b, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		if (!use_lower_bound_caching_) { return 0; }

		auto& hashmap = cache_[data.Size()];
		auto iter = FindIterator(data, b);// hashmap.find(data);

		if (iter == hashmap.end()) { return 0; }

		//compute the misclassification lower bound by considering that branches with more node/depth budgets 
		//  can only have less or equal misclassification than when using the prescribed number of nodes and depth
		int best_lower_bound = 0;
		for (CacheEntry& entry : iter->second) {
			if (num_nodes <= entry.GetNodeBudget() && depth <= entry.GetDepthBudget()
				&& entry.GetDiscriminationBudget() == b.GetDiscriminationBudget()) { // TODO reason about unequal budgets
				int local_lower_bound = entry.GetLowerBound();
				best_lower_bound = std::max(best_lower_bound, local_lower_bound);
			}
		}
		return best_lower_bound;
	}

	void DatasetCache::UpdateDiscrimationBudget(const Branch& org_branch, Branch& this_branch, BinaryData& data, const Branch& branch, int depth, int num_nodes, int upper_bound) {
		runtime_assert(1 == 2);

		// TODO implement
	}

	const DiscriminationBudget DatasetCache::RetrieveBestBudgetBounds(BinaryData& data, const Branch& org_branch, const Branch& branch, int depth, int num_nodes, int upper_bound) {
		runtime_assert(1 == 2);

		// TODO implement
	}

	int DatasetCache::NumEntries() const {
		size_t count = 0;
		for (auto& c : cache_) {
			count += c.size();
		}
		return int(count);
	}

	void DatasetCache::DisableLowerBounding() {
		use_lower_bound_caching_ = false;
	}

	void DatasetCache::DisableOptimalCaching() {
		use_optimal_caching_ = false;
	}

	void DatasetCache::InvalidateStoredIterators(BinaryData& data) {
		stored_iterators_[data.Size()].clear();
	}

	std::unordered_map<BinaryData, std::vector<CacheEntry>, DatasetHashFunction, DatasetEquality>::iterator DatasetCache::FindIterator(BinaryData& data, const Branch& branch) {
		for (PairIteratorBranch& p : stored_iterators_[data.Size()]) {
			if (p.branch == branch) { return p.iter; }
		}

		if (!data.IsHashSet()) { data.SetHash(DatasetHashFunction::ComputeHashForData(data)); }

		auto& hashmap = cache_[data.Size()];
		auto iter = hashmap.find(data);

		PairIteratorBranch hehe;
		hehe.branch = branch;
		hehe.iter = iter;

		if (stored_iterators_[data.Size()].size() == 2) { stored_iterators_[data.Size()].pop_back(); }
		stored_iterators_[data.Size()].push_front(hehe);
		return iter;
	}


}