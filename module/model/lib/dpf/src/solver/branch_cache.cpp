/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/branch_cache.h"

namespace DPF {

	BranchCache::BranchCache(int max_branch_length) :
		cache_(size_t(max_branch_length) + 1),
		use_lower_bound_caching_(USE_CACHE),
		use_optimal_caching_(USE_CACHE),
		use_budget_caching_(USE_CACHE) { }

	bool BranchCache::IsOptimalAssignmentCached(BinaryData&, const Branch& branch, int depth, int num_nodes, int upper_bound) {
		runtime_assert(depth <= num_nodes);

		if (!use_optimal_caching_) { return false; }

		auto& hashmap = cache_[branch.Depth()];
		auto iter = hashmap.find(branch);

		if (iter == hashmap.end()) { return false; }

		for (CacheEntry& entry : iter->second) {
			if (entry.GetNodeBudget() == num_nodes && entry.GetDepthBudget() == depth
				&& entry.GetDiscriminationBudget() >= branch.GetDiscriminationBudget()
				&& entry.GetUpperBound() >= upper_bound) {
				return entry.IsOptimal();
			}
		}
		return false;
	}

	void BranchCache::StoreOptimalBranchAssignment(BinaryData& data, const Branch& branch, std::shared_ptr<AssignmentContainer> optimal_solutions_c, int depth, int num_nodes, int upper_bound) {
		runtime_assert(depth <= num_nodes && num_nodes > 0);

		if (!use_optimal_caching_) { return; }

		optimal_solutions_c->RemoveTempData();

		auto& hashmap = cache_[branch.Depth()];
		auto iter_vector_entry = hashmap.find(branch);
		const int sol_num_nodes = optimal_solutions_c->NumNodes();
		int optimal_node_depth = std::min(depth, num_nodes); //this is an estimate of the depth, it could be lower actually. We do not consider lower for simplicity, but it would be good to consider it as well.
		auto disc_budget = branch.GetDiscriminationBudget();
		if (!optimal_solutions_c->GetPruned())
			disc_budget = DiscriminationBudget(-1.0, 1.0);

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry> vector_entry;
			for (int node_budget = sol_num_nodes; node_budget <= num_nodes; node_budget++) {
				for (int depth_budget = optimal_node_depth; depth_budget <= std::min(depth, node_budget); depth_budget++) {
					CacheEntry entry(depth_budget, node_budget, upper_bound, disc_budget, optimal_solutions_c, optimal_solutions_c->GetBestBounds());
					vector_entry.push_back(entry);
				}
			}
			cache_[branch.Depth()].insert(std::pair<Branch, std::vector<CacheEntry> >(branch, vector_entry));
		} else {
			//this sol is valid for size=[opt.NumNodes, num_nodes] and depths d=min(size, depth)

			//now we need to see if other node budgets have been seen before. 
			//For each budget that has been seen, update it;
			std::vector<std::vector<bool> > budget_seen(size_t(num_nodes) + 1, std::vector<bool>(depth + 1, false));
			for (CacheEntry& entry : iter_vector_entry->second) {
				//todo enable this here! //runtime_assert(optimal_node.Misclassifications() >= entry.GetLowerBound() || optimal_node.NumNodes() > entry.GetNodeBudget());

				//I believe it rarely happens that we receive a solution with less nodes than 'num_nodes', but it is possible
				if (sol_num_nodes <= entry.GetNodeBudget() && entry.GetNodeBudget() <= num_nodes
					&& optimal_node_depth <= entry.GetDepthBudget() && entry.GetDepthBudget() <= depth) {
					/*if (!(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_node.Misclassifications())) {
						std::cout << "opt node: " << optimal_node.NumNodes() << ", " << optimal_node.misclassification_score << "\n";
						std::cout << "\tnum nodes: " << num_nodes << "\n";
						std::cout << entry.GetNodeBudget() << ", " << entry.GetOptimalValue() << "\n";
					} // TODO fix this
					runtime_assert(!entry.IsOptimal() || entry.GetOptimalValue() == optimal_node.Misclassifications());*/

					if (entry.GetDiscriminationBudget() == branch.GetDiscriminationBudget()
						&& entry.GetUpperBound() == upper_bound) {
						budget_seen[entry.GetNodeBudget()][entry.GetDepthBudget()] = true;
						if (!entry.IsOptimal()) { entry.SetOptimalSolutions(optimal_solutions_c); }
					}

					runtime_assert(entry.GetDepthBudget() <= entry.GetNodeBudget()); //fix the case when it turns out that more nodes do not give a better result...e.g., depth 4 and num nodes 4, but a solution with three nodes found...that solution is then optimal for depth 3 as well...need to update but lazy now
					int g = 0;
					g++;
				}
			}
			//create entries for those which were not seen
			//note that most of the time this loop only does one iteration since usually using the full node budget gives better results
			for (int node_budget = sol_num_nodes; node_budget <= num_nodes; node_budget++) {
				for (int depth_budget = optimal_node_depth; depth_budget <= std::min(depth, node_budget); depth_budget++) {
					if (!budget_seen[node_budget][depth_budget]) {
						CacheEntry entry(depth_budget, node_budget, upper_bound, disc_budget, optimal_solutions_c, optimal_solutions_c->GetBestBounds());
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
		//TODO fix this statement
		//runtime_assert(RetrieveOptimalAssignment(data, branch, depth, num_nodes).Misclassifications() == optimal_node.Misclassifications());
	}

	void BranchCache::TransferAssignmentsForEquivalentBranches(const BinaryData&, const Branch& branch_source, const BinaryData&, const Branch& branch_destination) {
		if (!use_lower_bound_caching_) { return; }

		if (branch_source == branch_destination) { return; }

		auto& hashmap = cache_[branch_source.Depth()];
		auto iter_source = hashmap.find(branch_source);
		auto iter_destination = hashmap.find(branch_destination);

		runtime_assert(!use_optimal_caching_ || iter_source != hashmap.end());//I believe the method will not be called if branch_source is empty, todo check
		if (iter_source == hashmap.end()) { return; }

		if (iter_destination == hashmap.end()) //if the branch has never been seen before, create a new entry for it and copy everything into it
		{
			std::vector<CacheEntry> vector_entry = iter_source->second;
			cache_[branch_destination.Depth()].insert(std::pair<Branch, std::vector<CacheEntry> >(branch_destination, vector_entry));
		} else {
			for (CacheEntry& entry_source : iter_source->second) {
				//todo could be done more efficiently
				bool should_add = true;
				for (CacheEntry& entry_destination : iter_destination->second) {
					if (entry_source.GetDepthBudget() == entry_destination.GetDepthBudget() &&
						entry_source.GetNodeBudget() == entry_destination.GetNodeBudget() &&
						entry_source.GetDiscriminationBudget() == entry_destination.GetDiscriminationBudget() &&
						entry_source.GetUpperBound() == entry_destination.GetUpperBound()) {
						should_add = false;
						//if the source entry is strictly better than the destination entry, replace it
						if (entry_source.IsOptimal() && !entry_destination.IsOptimal() || entry_source.GetLowerBound() > entry_destination.GetLowerBound()) {
							entry_destination = entry_source;
							break;
						}
					}
				}
				if (should_add) { iter_destination->second.push_back(entry_source); }
			}
		}
	}

	std::shared_ptr<AssignmentContainer> BranchCache::RetrieveOptimalAssignment(BinaryData& data, const Branch& branch, int depth, int num_nodes, int upper_bound) {
		auto& hashmap = cache_[branch.Depth()];
				
		auto iter = hashmap.find(branch);
		if (iter == hashmap.end()) { return std::shared_ptr<AssignmentContainer>(nullptr); }

		for (CacheEntry& entry : iter->second) {
			if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes && entry.IsOptimal() &&
				entry.GetDiscriminationBudget() >= branch.GetDiscriminationBudget() &&
				entry.GetUpperBound() >= upper_bound) {
				return entry.GetOptimalSolution();
			}
		}
		return std::shared_ptr<AssignmentContainer>(nullptr);
	}

	void BranchCache::UpdateLowerBound(BinaryData&, const Branch& branch, int lower_bound, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		if (!use_lower_bound_caching_) { return; }

		auto& hashmap = cache_[branch.Depth()];
		auto iter_vector_entry = hashmap.find(branch);

		//if the branch has never been seen before, create a new entry for it
		if (iter_vector_entry == hashmap.end()) {
			std::vector<CacheEntry> vector_entry(1, CacheEntry(depth, num_nodes, INT32_MAX)); // TODO what discrimination budget to set here?
			vector_entry[0].UpdateLowerBound(lower_bound);
			cache_[branch.Depth()].insert(std::pair<Branch, std::vector<CacheEntry> >(branch, vector_entry));
		} else {
			//now we need to see if this node node_budget has been seen before. 
			//If it was seen, update it; otherwise create a new entry
			bool found_corresponding_entry = false;
			for (CacheEntry& entry : iter_vector_entry->second) {
				//If the new lower bound is found with more relaxed discrimination budget, 
				// yet it has a higher lower-bound, update this entry
				if (entry.GetDepthBudget() == depth && entry.GetNodeBudget() == num_nodes &&
					branch.GetDiscriminationBudget() >= entry.GetDiscriminationBudget()) {
					if(!entry.IsOptimal() && lower_bound >= entry.GetLowerBound())
						entry.UpdateLowerBound(lower_bound);
					found_corresponding_entry = true;
					break;
				}
			}

			if (!found_corresponding_entry) {
				CacheEntry entry(depth, num_nodes, INT32_MAX);
				entry.UpdateLowerBound(lower_bound);
				iter_vector_entry->second.push_back(entry);
			}
		}
	}

	int BranchCache::RetrieveLowerBound(BinaryData&, const Branch& branch, int depth, int num_nodes) {
		runtime_assert(depth <= num_nodes);

		if (!use_lower_bound_caching_) { return 0; }

		auto& hashmap = cache_[branch.Depth()];
		auto iter = hashmap.find(branch);

		if (iter == hashmap.end()) { return 0; }

		//compute the misclassification lower bound by considering that branches with more node/depth budgets 
		//  can only have less or equal misclassification than when using the prescribed number of nodes and depth
		int best_lower_bound = 0;
		for (CacheEntry& entry : iter->second) {
			if (num_nodes <= entry.GetNodeBudget() && depth <= entry.GetDepthBudget() && 
				entry.GetDiscriminationBudget() >= branch.GetDiscriminationBudget()) {
				int local_lower_bound = entry.GetLowerBound();
				best_lower_bound = std::max(best_lower_bound, local_lower_bound);
			}
		}
		return best_lower_bound;
	}

	void BranchCache::UpdateDiscrimationBudget(const Branch& org_branch, Branch& this_branch, BinaryData& data, const Branch& branch, int depth, int num_nodes, int upper_bound) {
		runtime_assert(depth <= num_nodes);
		if (!use_budget_caching_) { return; }

		auto& hashmap = cache_[branch.Depth()];
		auto iter = hashmap.find(branch);

		if (iter == hashmap.end()) { return; }

		DiscriminationBudget best_budget = this_branch.GetDiscriminationBudget();
		for (CacheEntry& entry : iter->second) {
			if (num_nodes <= entry.GetNodeBudget() && depth <= entry.GetDepthBudget()
				&& entry.GetDiscriminationBudget() >= branch.GetDiscriminationBudget()
				&& entry.GetUpperBound() >= upper_bound) {
				best_budget.Tighten(org_branch.GetDiscriminationBudget(), entry.GetBestBudget());
			}
		}
		this_branch.SetDiscriminationBudget(best_budget);

	}

	const DiscriminationBudget BranchCache::RetrieveBestBudgetBounds(BinaryData& data, const Branch& org_branch, const Branch& branch, int depth, int num_nodes, int upper_bound) {
		runtime_assert(depth <= num_nodes);

		if (!use_budget_caching_) { return DiscriminationBudget::nonRestrictedBudget; }

		auto& hashmap = cache_[branch.Depth()];
		auto iter = hashmap.find(branch);

		if (iter == hashmap.end()) { return DiscriminationBudget::nonRestrictedBudget; }

		//compute the misclassification lower bound by considering that branches with more node/depth budgets 
		//  can only have less or equal misclassification than when using the prescribed number of nodes and depth
		DiscriminationBudget best_budget = DiscriminationBudget::nonRestrictedBudget;
		for (CacheEntry& entry : iter->second) {
			if (num_nodes <= entry.GetNodeBudget() && depth <= entry.GetDepthBudget()
				&& entry.GetDiscriminationBudget() >= branch.GetDiscriminationBudget()
				&& entry.GetUpperBound() >= upper_bound) {
				best_budget.Tighten(org_branch.GetDiscriminationBudget(), entry.GetBestBudget());
			}
		}
		return best_budget;
	}

	int BranchCache::NumEntries() const {
		size_t count = 0;
		for (auto& c : cache_) {
			count += c.size();
		}
		return int(count);
	}

	void BranchCache::DisableLowerBounding() {
		use_lower_bound_caching_ = false;
	}

	void BranchCache::DisableOptimalCaching() {
		use_optimal_caching_ = false;
	}

}