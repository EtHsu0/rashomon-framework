/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "solver/solver.h"

#define DEBUG_PRINT false

void PrintIndent(int depth) {
	for (int i = 0; i < depth; i++) std::cout << "  ";
}

namespace DPF {

	Solver::Solver(ParameterHandler& solver_parameters) :
		parameters(solver_parameters),
		return_pareto_front(parameters.GetStringParameter("mode")=="pareto"),
		verbose_(parameters.GetBooleanParameter("verbose")),
		cache(0), terminal_solver1(nullptr), terminal_solver2(nullptr), similarity_lower_bound_computer(nullptr)
	{
		discrimination_cutoff_value = parameters.GetFloatParameter("stat-test-value");
		minimum_leaf_node_size = int(parameters.GetIntegerParameter("min-leaf-node-size"));
		std::string file = parameters.GetStringParameter("file");
		if (file != "manual") {
			binary_data = FileReader::ReadDataDL(file,
				int(parameters.GetIntegerParameter("num-instances")),
				int(parameters.GetIntegerParameter("max-num-features")),
				int(parameters.GetIntegerParameter("duplicate-factor")));
			if (verbose_) {
				std::cout << "Succesfully read " << file << std::endl;
				binary_data->PrintStats();
			}
			Reset();
		} else {
			binary_data = nullptr;
			num_features = 0;
			num_labels_ = 2;
			sparsity_coefficient = 0.0;
		}		
	}

	Solver::~Solver() {
		if (cache != nullptr) delete cache;
		if (terminal_solver1 != nullptr) delete terminal_solver1;
		if (terminal_solver2 != nullptr) delete terminal_solver2;
		if (binary_data != nullptr) binary_data->DestroyData();
		delete binary_data;
		if (similarity_lower_bound_computer != nullptr) delete similarity_lower_bound_computer;
	}

	void Solver::Reset() {
		runtime_assert(binary_data != nullptr);
		if(cache != nullptr)
			delete cache;
		SplitTrainTestData();
		data_summary = DataSummary(train_data);
		num_features = binary_data->NumFeatures();
		sparsity_coefficient = std::round(parameters.GetFloatParameter("sparsity") * train_data.Size());
		if (parameters.GetStringParameter("cache-type") == "dataset") {
			cache = new DatasetCache(train_data.Size());
		} else if (parameters.GetStringParameter("cache-type") == "branch") {
			cache = new BranchCache(100);
		} else {
			std::cout << "Parameter error: unknown cache type: " << parameters.GetStringParameter("cache-type") << "\n";
			runtime_assert(1 == 2);
		}
		if (terminal_solver1 != nullptr)
			delete terminal_solver1;
		if (terminal_solver2 != nullptr)
			delete terminal_solver2;
		terminal_solver1 = new TerminalSolver(this, num_features);
		terminal_solver2 = new TerminalSolver(this, num_features);
		if (similarity_lower_bound_computer != nullptr)
			delete similarity_lower_bound_computer;
		similarity_lower_bound_computer = new SimilarityLowerBoundComputer(100, 100, train_data.Size());
		if (parameters.GetBooleanParameter("similarity-lower-bound") == false) { similarity_lower_bound_computer->Disable(); }
		stats = Statistics();
	}

	void Solver::SetData(const BinaryData* data) {
		this->binary_data = data->GetDeepCopy();
	}

	void Solver::SplitTrainTestData() {
		if (parameters.GetFloatParameter("train-test-split") <= DBL_EPSILON) {
			train_data = *binary_data;
		} else {
			binary_data->TrainTestSplitData(parameters.GetFloatParameter("train-test-split"), train_data, test_data);
		}
	}

	const SolverResult Solver::HyperSolve() {
		runtime_assert(parameters.GetStringParameter("mode") == "hyper");
		runtime_assert(parameters.GetFloatParameter("train-test-split") > 0);
		bool verbose = parameters.GetBooleanParameter("verbose");
		int max_nodes = parameters.GetIntegerParameter("max-num-nodes");
		int best_misclassification = INT32_MAX;
		double best_disc = 1.0;
		int best_num_nodes = 0;
		const double cut_off = parameters.GetFloatParameter("stat-test-value");
		for (int i = 0; i <= max_nodes; i++) {
			ParameterHandler parameterse(this->parameters);
			parameters.SetIntegerParameter("max-num-nodes", i);
			parameters.SetStringParameter("file", "manual");
			parameters.SetStringParameter("mode", "best");
			std::vector<DPF::Performance> performances;
			for (int r = 0; r < 10; r++) {
				Solver solver(parameters);
				solver.SetData(&(this->train_data));
				solver.Reset();
				clock_t clock_before_solve = clock();
				const auto result = solver.Solve();
				runtime_assert(result.performances.size() == 1);
				performances.push_back(result.performances[0]);
			}
			auto perf = DPF::Performance::GetAverage(performances);
			if (std::abs(perf.test_discrimination) <= cut_off) {
				if (perf.test_misclassifications < best_misclassification) {
					best_misclassification = perf.test_misclassifications;
					best_num_nodes = i;
					best_disc = std::abs(perf.test_discrimination);
				} else if (perf.test_misclassifications == best_misclassification && perf.test_discrimination < best_disc) {
					best_misclassification = perf.test_misclassifications;
					best_num_nodes = i;
					best_disc = std::abs(perf.test_discrimination);
				}
			} else if (best_misclassification == INT32_MAX && std::abs(perf.test_discrimination) <= best_disc) {
				best_num_nodes = i;
				best_disc = std::abs(perf.test_discrimination);
			}
		}
		if (verbose) {
			std::cout << std::endl << "Finished hyper parameter search. Best number of nodes: " << best_num_nodes << std::endl;
		}
		parameters.SetIntegerParameter("max-num-nodes", best_num_nodes);
		return Solve();
	}

	const SolverResult Solver::Solve() {
		stopwatch_.Initialise(parameters.GetFloatParameter("time"));

		Branch root_branch;
		int misclassification_upper_bound = int(parameters.GetIntegerParameter("upper-bound"));
		auto best_solutions = CreateLeafNodeDescriptions(train_data, Branch(), misclassification_upper_bound);
		InternalNodeDescription best_solution;
		if (best_solutions->Size() > 0) {
			auto& trivial_solution = (*best_solutions)[0];
			if (verbose_) std::cout << "Initial trivial solution: " << trivial_solution.GetPartialSolution() << std::endl;
			misclassification_upper_bound = trivial_solution.GetObjectiveScore(sparsity_coefficient);
			best_solution = trivial_solution;
		}

		int min_num_nodes = int(parameters.GetIntegerParameter("max-num-nodes"));
		if (parameters.GetBooleanParameter("all-trees") || sparsity_coefficient > 0 ) { min_num_nodes = 1; }
		

		for (int num_nodes = min_num_nodes; num_nodes <= parameters.GetIntegerParameter("max-num-nodes"); num_nodes++) {
			if (!stopwatch_.IsWithinTimeLimit()) { break; }
			if (verbose_) std::cout << "num nodes: " << num_nodes << " " << stopwatch_.TimeElapsedInSeconds() << "s" << std::endl;

			int max_depth = std::min(int(parameters.GetIntegerParameter("max-depth")), num_nodes);
			auto new_solutions = SolveSubtree(
				train_data,
				root_branch,
				max_depth,
				num_nodes,
				misclassification_upper_bound
			);

			for (const auto& node : new_solutions->solutions) {
				runtime_assert(node.IsFeasible());
				if(SatisfiesConstraint(node, root_branch) && node.GetObjectiveScore(sparsity_coefficient) < misclassification_upper_bound) {
					if(!return_pareto_front)
						misclassification_upper_bound = node.GetObjectiveScore(sparsity_coefficient);
					best_solution = node;
				}
				best_solutions->Add(node);
			}
			best_solutions->Filter(data_summary);
			//break;
		}

		if (verbose_) std::cout << "Reconstruct optimal tree" << std::endl;
		SolverResult result;
		result.is_proven_optimal = stopwatch_.IsWithinTimeLimit();
		if (parameters.GetStringParameter("mode") == "best") {
			best_solutions = std::make_shared<AssignmentContainer>(false, parameters.GetIntegerParameter("max-num-nodes"));
			best_solutions->Add(best_solution);
		} else {
			if (verbose_) std::cout << "Found " << best_solutions->Size() << " solutions." << std::endl;
		}
		result.solutions = best_solutions;
		if (result.is_proven_optimal) {
			for (auto& sol : best_solutions->solutions) {
				auto tree = ConstructOptimalTree(sol, train_data, root_branch, std::min(int(parameters.GetIntegerParameter("max-depth")), sol.NumNodes()), sol.NumNodes());
				result.trees.push_back(tree);
				result.is_proven_optimal = stopwatch_.IsWithinTimeLimit();
				auto performance = Performance::ComputePerformance(tree.get(), train_data, test_data);
				result.performances.push_back(performance);
				if (performance.train_misclassifications == sol.GetMisclassifications() &&
					std::abs(std::abs(performance.train_discrimination) - sol.GetBestDiscrimination()) <= 1e-4 ) {
					if (verbose_) std::cout << "Tree misclassification score has been verified!\n";
				} else {
					if (result.is_proven_optimal) {
						std::cout << "Verification failed. Misclassification score: " << performance.train_misclassifications << ". Discrimination: " << performance.train_discrimination << "." << std::endl;
						std::cout << "Expected values according to the algorithm: Misclassification score: " << sol.GetMisclassifications() << ". Discrimination: " << sol.GetBestDiscrimination() << std::endl;
						std::cout << "Please report this issue to Koos van der Linden, J.G.M.vanderLinden@tudelft.nl\n";
					}
				}

			}
		}

		if (verbose_) {
			std::cout << "Cache entries: " << cache->NumEntries() << std::endl;
			std::cout << "Cache optimal hits: " << stats.num_cache_hit_optimality << std::endl;
			std::cout << "Cache non zero lower bounds: " << stats.num_cache_hit_nonzero_bound << std::endl << std::endl;
			for (int i = 0; i < max_depth; i++) {
				std::cout << "Node stats for depth=" << i << std::endl;
				std::cout << "Time spent merging in this layer: " << stats.time_merging_per_layer[i]
					<< " (" << std::setprecision(4) << (stats.time_merging_per_layer[i] * 100 / stats.time_merging) << "%)" << std::endl;

				std::cout << "   Partial solution candidates: " << stats.num_partial_solution_candidates[i] << "\n";
				if (stats.num_partial_solution_candidates[i] == 0) break;
				size_t prev = stats.num_partial_solution_candidates[i];
				if (i == 0) {
					std::cout << "   Partial solution candidates relaxed fair: " << stats.num_partial_solution_candidates_relaxed_fair[i]
						<< " (" << (stats.num_partial_solution_candidates_relaxed_fair[i] * 100 / stats.num_partial_solution_candidates[i]) << "%)\n";
					prev = stats.num_partial_solution_candidates_relaxed_fair[i];
					if (prev == 0) continue;
				}
				std::cout << "   Partial solution candidates within upper bound: " << stats.num_partial_solution_candidates_within_upperbound[i]
					<< " (" << (stats.num_partial_solution_candidates_within_upperbound[i] * 100 / prev) << "%)\n";
				if (stats.num_partial_solution_candidates_within_upperbound[i] == 0) continue;
				std::cout << "   Partial solution unique: " << stats.num_partial_solution_candidates_unique[i]
					<< " (" << (stats.num_partial_solution_candidates_unique[i] * 100 / stats.num_partial_solution_candidates_within_upperbound[i]) << "%)\n";
				if (stats.num_partial_solution_candidates_unique[i] == 0) continue;
				std::cout << "   Partial solution candidates possibly fair: " << stats.num_partial_solution_candidates_possibly_fair[i]
					<< " (" << (stats.num_partial_solution_candidates_possibly_fair[i] * 100 / stats.num_partial_solution_candidates_unique[i]) << "%)\n";
				if (stats.num_partial_solution_candidates_possibly_fair[i] == 0) continue;
				std::cout << "   Partial solution actually added to pareto-front: " << stats.num_computed_non_dom_nodes[i]
					<< " (" << (stats.num_computed_non_dom_nodes[i] * 100 / stats.num_partial_solution_candidates_possibly_fair[i]) << "%)\n";
			}

			std::cout << "Terminal time: " << stats.time_in_terminal_node << "\n";
			std::cout << "Terminal calls: " << stats.num_terminal_nodes_with_node_budget_one + stats.num_terminal_nodes_with_node_budget_two + stats.num_terminal_nodes_with_node_budget_three << "\n";
			std::cout << "\tTerminal 1 node: " << stats.num_terminal_nodes_with_node_budget_one << "\n";
			std::cout << "\tTerminal 2 node: " << stats.num_terminal_nodes_with_node_budget_two << "\n";
			std::cout << "\tTerminal 3 node: " << stats.num_terminal_nodes_with_node_budget_three << "\n";
		}

		return result;
	}

	std::shared_ptr<AssignmentContainer> Solver::SolveSubtree(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound) {
		runtime_assert(0 <= max_depth && max_depth <= num_nodes);
		bool root_node = branch.Depth() == 0;
		if (!stopwatch_.IsWithinTimeLimit()) { return std::make_shared<AssignmentContainer>(root_node, num_nodes); }
		if (upper_bound < 0) { return std::make_shared<AssignmentContainer>(root_node, num_nodes); }
		if (max_depth == 0 || num_nodes == 0) { 
			auto leaf_nodes = CreateLeafNodeDescriptions(data, branch, upper_bound);
			return leaf_nodes;
		}

		// Check Cache
		{
			auto results = cache->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes, upper_bound);
			if (results.get() != nullptr) {
				stats.num_cache_hit_optimality++;
				size_t initial_size = results->Size();
				results->FilterOnUpperbound(upper_bound);
				results->FilterOnDiscriminationBounds(branch, data_summary, discrimination_cutoff_value);
				if (DEBUG_PRINT) {
					PrintIndent(branch.Depth());
					std::cout << "D=" << branch.Depth() << " Found " << results->Size() << " solutions in the cache." << std::endl;
				}
				if(initial_size > results->Size())
					results->Filter(data_summary);
				else
					results->UpdateBestBounds(data_summary);
				return results;
			}
		}

		// Update the cache using the similarity-based lower bound 
		// If an optimal solution was found in the process, return it.
		bool updated_optimal_solution = UpdateCacheUsingSimilarity(data, branch, max_depth, num_nodes, upper_bound);
		if (updated_optimal_solution) {
			// Copied from above. TODO reafactor into non-duplicate code
			auto results = cache->RetrieveOptimalAssignment(data, branch, max_depth, num_nodes, upper_bound);
			if (results.get() != nullptr) {
				stats.num_cache_hit_optimality++;
				size_t initial_size = results->Size();
				results->FilterOnUpperbound(upper_bound);
				results->FilterOnDiscriminationBounds(branch, data_summary, discrimination_cutoff_value);
				if (DEBUG_PRINT) {
					PrintIndent(branch.Depth());
					std::cout << "D=" << branch.Depth() << " Found " << results->Size() << " solutions in the cache." << std::endl;
				}
				if (initial_size > results->Size())
					results->Filter(data_summary);
				else
					results->UpdateBestBounds(data_summary);
				return results;
			}
		}

		//Check LB >= UB
		int lower_bound = cache->RetrieveLowerBound(data, branch, max_depth, num_nodes);
		if (lower_bound > 0) stats.num_cache_hit_nonzero_bound++;
		if (lower_bound > upper_bound) {
			if (DEBUG_PRINT) {
				PrintIndent(branch.Depth());
				std::cout << "D=" << branch.Depth() << " Cut out because LB > UB" << std::endl;
			}
			return std::make_shared<AssignmentContainer>(root_node, num_nodes);
		}

		// Use the specialised algorithm for small trees
		if (SPECIAL_D2 && IsTerminalNode(max_depth, num_nodes)) { return SolveTerminalNode(data, branch, max_depth, num_nodes, upper_bound); }


		return SolveSubtreeGeneralCase(data, branch, max_depth, num_nodes, upper_bound);
	}

	std::shared_ptr<AssignmentContainer> Solver::SolveTerminalNode(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound) {
		runtime_assert(max_depth <= 2 && 1 <= num_nodes && num_nodes <= 3 && max_depth <= num_nodes);
		runtime_assert(num_nodes != 3 || !cache->IsOptimalAssignmentCached(data, branch, 2, 3, upper_bound));
		runtime_assert(num_nodes != 2 || !cache->IsOptimalAssignmentCached(data, branch, 2, 2, upper_bound));
		runtime_assert(num_nodes != 1 || !cache->IsOptimalAssignmentCached(data, branch, 1, 1, upper_bound));

		stats.num_terminal_nodes_with_node_budget_one += (num_nodes == 1);
		stats.num_terminal_nodes_with_node_budget_two += (num_nodes == 2);
		stats.num_terminal_nodes_with_node_budget_three += (num_nodes == 3);

		clock_t clock_start = clock();
		//select the solver which is already contains frequency counts that are closest to the data
		TerminalResults results;
		int diff1 = terminal_solver1->ProbeDifference(data);
		int diff2 = terminal_solver2->ProbeDifference(data);
		if (diff1 < diff2) {
			results = terminal_solver1->Solve(data, branch, num_nodes, upper_bound);
		} else {
			results = terminal_solver2->Solve(data, branch, num_nodes, upper_bound);
		}		
		stats.time_in_terminal_node += double(clock() - clock_start) / CLOCKS_PER_SEC;

		//since the specialised algorithm computes trees of size 1, 2, 3 for depth 2
		//	we store all these results in the cache to avoid possibly recomputing later

		if (!cache->IsOptimalAssignmentCached(data, branch, 1, 1, upper_bound)) {
			auto& one_node_solutions = results.one_node_solutions;
			one_node_solutions->Filter(data_summary);
			runtime_assert(one_node_solutions->NumNodes() <= 1);
			if (one_node_solutions->IsFeasible()) {
				cache->StoreOptimalBranchAssignment(data, branch, one_node_solutions, 1, 1, upper_bound);
			} else {
				cache->UpdateLowerBound(data, branch, upper_bound + 1, 1, 1);
			}
		}

		if (num_nodes >= 2 && !cache->IsOptimalAssignmentCached(data, branch, 2, 2, upper_bound)) {
			auto& two_nodes_solutions = results.two_nodes_solutions;
			two_nodes_solutions->Filter(data_summary);
			runtime_assert(two_nodes_solutions->NumNodes() <= 2);
			if(two_nodes_solutions->IsFeasible()) {
				cache->StoreOptimalBranchAssignment(data, branch, two_nodes_solutions, 2, 2, upper_bound);
			} else {
				cache->UpdateLowerBound(data, branch, upper_bound + 1, 2, 2);
			}
		}

		if (num_nodes == 3 && !cache->IsOptimalAssignmentCached(data, branch, 2, 3, upper_bound)) {
			auto& three_nodes_solutions = results.three_nodes_solutions;
			three_nodes_solutions->Filter(data_summary);
			runtime_assert(three_nodes_solutions->NumNodes() <= 3);
			if(three_nodes_solutions->IsFeasible()) {
				cache->StoreOptimalBranchAssignment(data, branch, three_nodes_solutions, 2, 3, upper_bound);
			} else {
				cache->UpdateLowerBound(data, branch, upper_bound + 1, 2, 3);
			}
		}

		similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);

		std::shared_ptr<AssignmentContainer> result;
		if (num_nodes == 1) result = std::make_shared<AssignmentContainer>(*(results.one_node_solutions));
		else if(num_nodes == 2) result = std::make_shared<AssignmentContainer>(*(results.two_nodes_solutions));
		else result = std::make_shared<AssignmentContainer>(*(results.three_nodes_solutions));
		result->FilterOnUpperbound(upper_bound);
		result->UpdateBestBounds(data_summary);
		return result;
	}

	std::shared_ptr<AssignmentContainer> Solver::SolveSubtreeGeneralCase(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound) {
		runtime_assert(max_depth <= num_nodes);

		const int max_size_subtree = std::min((1 << (max_depth - 1)) - 1, num_nodes - 1); //take the minimum between a full tree of max_depth or the number of nodes - 1
		const int min_size_subtree = num_nodes - 1 - max_size_subtree;
		const int org_upper_bound = upper_bound;
		const int initial_upper_bound = upper_bound;
		auto solutions = CreateLeafNodeDescriptions(data, branch, upper_bound);
		solutions->num_nodes = num_nodes; // TODO different values for the sub-solutions

		std::unique_ptr<FeatureSelectorAbstract> feature_selector;
		if(parameters.GetStringParameter("feature-ordering") == "in-order")
			feature_selector = std::make_unique<FeatureSelectorInOrder>(num_features);
		else if(parameters.GetStringParameter("feature-ordering") == "gini")
			feature_selector = std::make_unique<FeatureSelectorInOrder>(num_features);
		else { std::cout << "Unknown feature ordering strategy!\n"; exit(1); }
		feature_selector->Initialize(data);

		int current_depth = branch.Depth();


		while (feature_selector->AreThereAnyFeaturesLeft()) {
			if (!stopwatch_.IsWithinTimeLimit()) break;
			int feature = feature_selector->PopNextFeature();
			if (branch.HasBranchedOnFeature(feature)) continue;
			if (DEBUG_PRINT && current_depth <= 1) {
				PrintIndent(current_depth);
				std::cout << "D=" << branch.Depth() << " Branch on feature " << feature << " (" << max_depth << ", " << num_nodes << ")" << std::endl;
			}
			
			BinaryData left_data;
			BinaryData right_data;
			data.SplitData(feature, left_data, right_data);
			if (left_data.Size() < minimum_leaf_node_size || right_data.Size() < minimum_leaf_node_size) continue;

			BinaryData* large_data = &left_data;
			BinaryData* small_data = &right_data;
			
			Branch left_branch = Branch::LeftChildBranch(branch, feature, GetDiscriminationBudget(*small_data, branch));
			Branch right_branch = Branch::RightChildBranch(branch, feature, GetDiscriminationBudget(*large_data, branch));

			bool switch_left_right = false;
			if (small_data->Size() < large_data->Size()) {
				std::swap(small_data, large_data);
				std::swap(left_branch, right_branch);
				switch_left_right = true;
			}

			for (int left_subtree_size = min_size_subtree; left_subtree_size <= max_size_subtree; left_subtree_size++) {
				int right_subtree_size = num_nodes - left_subtree_size - 1; //the '-1' is necessary since using the parent node counts as a node
				int left_depth = std::min(max_depth - 1, left_subtree_size);
				int right_depth = std::min(max_depth - 1, right_subtree_size);

				Branch left_branch2 = left_branch;
				int right_lower_bound = cache->RetrieveLowerBound(*small_data, right_branch, right_depth, right_subtree_size);
				if (right_lower_bound > 0) stats.num_cache_hit_nonzero_bound++;
				int left_upper_bound = upper_bound - right_lower_bound;
				int left_lower_bound = cache->RetrieveLowerBound(*large_data, left_branch, left_depth, left_subtree_size);
				if (left_lower_bound > 0) stats.num_cache_hit_nonzero_bound++;
				int right_upper_bound = upper_bound - left_lower_bound;
#if USE_PRUNE
					cache->UpdateDiscrimationBudget(branch, left_branch2, *small_data, right_branch, right_depth, right_subtree_size, right_upper_bound);
					similarity_lower_bound_computer->UpdateDiscrimationBudget(branch, left_branch2, *small_data, right_branch, data_summary, right_depth, right_subtree_size, right_upper_bound, cache);
					if (!left_branch2.GetDiscriminationBudget().IsFeasible()) continue;
#endif

				auto left_solutions = SolveSubtree(
					*large_data,
					left_branch2,
					left_depth,
					left_subtree_size,
					left_upper_bound
				);

				if (!stopwatch_.IsWithinTimeLimit()) break;
				if (left_solutions->Size() == 0) continue;

				// Update upper bound and budget based on other subtree
				right_upper_bound = upper_bound - left_solutions->GetLeastMisclassifications();
				
				Branch right_branch2 = right_branch;
#if USE_PRUNE
				UpdateDiscriminationBudget(branch, right_branch2, left_solutions.get());
				if (!right_branch2.GetDiscriminationBudget().IsFeasible()) continue;
#endif

				auto right_solutions = SolveSubtree(
					*small_data,
					right_branch2,
					right_depth,
					right_subtree_size,
					right_upper_bound
				);

				if (!stopwatch_.IsWithinTimeLimit()) break;
				if (right_solutions->Size() == 0) continue;

				if (DEBUG_PRINT && current_depth <= 1) {
					for (int i = 0; i < current_depth; i++) std::cout << "  ";
					std::cout << "D=" << branch.Depth() << " Found " << left_solutions->Size() << " and " << right_solutions->Size() << ". Starting Merge" << std::endl;
				}
				if(switch_left_right)
					Merge<false>(feature, branch, right_branch2, left_branch2, upper_bound, right_solutions.get(), left_solutions.get(), solutions.get());
				else
					Merge<false>(feature, branch, left_branch2, right_branch2, upper_bound, left_solutions.get(), right_solutions.get(), solutions.get());
				
			}
			if (!stopwatch_.IsWithinTimeLimit()) break;
		}
		
		if (!return_pareto_front && upper_bound != INT32_MAX && upper_bound + 1 < initial_upper_bound) {
			solutions->FilterOnUpperbound(upper_bound + 1); // Plus one in order not to remove the optimal solution
		}

		solutions->Filter(data_summary);
		if (DEBUG_PRINT && current_depth <= 1) {
			for (int i = 0; i < current_depth; i++) std::cout << "  ";
			std::cout << "D=" << branch.Depth() << " Found " << solutions->Size() << "." << std::endl;
		}

		if (solutions->IsFeasible()) {
			cache->StoreOptimalBranchAssignment(data, branch, solutions, max_depth, num_nodes, org_upper_bound);
			similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);
		} else {
			cache->UpdateLowerBound(data, branch, upper_bound, max_depth, num_nodes);
			similarity_lower_bound_computer->UpdateArchive(data, branch, max_depth);
		}

		return solutions;
	}


#if BETTER_MERGE
	template<bool reconstruct>
	void Solver::Merge(int feature, const Branch& branch, const Branch& left_branch_, const Branch& right_branch_,
			int& upper_bound, AssignmentContainer* left_solutions, AssignmentContainer* right_solutions, AssignmentContainer* final_solutions, InternalTreeNode* tree_node) {
		if (left_solutions->Size() == 0 || right_solutions->Size() == 0) return;
		runtime_assert(reconstruct != (tree_node == nullptr));
		clock_t clock_start = clock();
		const int current_depth = branch.Depth();
		const int max_depth = int(parameters.GetIntegerParameter("max-depth"));
		size_t i = 0;
		const int initial_upper_bound = upper_bound;
		const size_t delta = 10000000;
		{
			size_t pruned = 0;
			Branch left_branch(left_branch_);
			Branch right_branch(right_branch_);
			while (left_solutions->Size() * right_solutions->Size() >= 10) { // TODO parameter tuning
				int left_upper_bound = upper_bound - right_solutions->GetLeastMisclassifications();
				int right_upper_bound = upper_bound - left_solutions->GetLeastMisclassifications();
				size_t initial_left_size = left_solutions->Size();
				if (right_solutions->Size() == 0) break;
				left_solutions->FilterOnUpperbound(left_upper_bound);
				UpdateDiscriminationBudget(branch, left_branch, right_solutions);
				left_solutions->FilterOnDiscriminationBounds(left_branch, data_summary, discrimination_cutoff_value);
				size_t new_left_size = left_solutions->Size();
				if (initial_left_size == new_left_size || new_left_size == 0) break;
				left_solutions->Filter(data_summary);
				left_upper_bound = left_solutions->GetLeastMisclassifications();
				right_upper_bound = upper_bound - left_upper_bound;
				pruned += initial_left_size - new_left_size;
				
				size_t initial_right_size = right_solutions->Size();
				right_solutions->FilterOnUpperbound(right_upper_bound);
				UpdateDiscriminationBudget(branch, right_branch, left_solutions);
				right_solutions->FilterOnDiscriminationBounds(right_branch, data_summary, discrimination_cutoff_value);
				size_t new_right_size = right_solutions->Size();
				if (initial_right_size == new_right_size || new_right_size == 0) break;
				right_solutions->Filter(data_summary);
				right_upper_bound = right_solutions->GetLeastMisclassifications();
				left_upper_bound = upper_bound - right_upper_bound;
				pruned += initial_right_size - new_right_size;
			}
		}
		if (left_solutions->Size() == 0 || right_solutions->Size() == 0) {
			double time_inc = double(clock() - clock_start) / CLOCKS_PER_SEC;
			stats.time_merging += time_inc;
			stats.time_merging_per_layer[current_depth] += time_inc;
			return;
		}
		size_t total = left_solutions->Size() * right_solutions->Size();
		stats.num_partial_solution_candidates[current_depth] += total;
		
		auto large_solution_set = left_solutions;
		auto small_solution_set = right_solutions;
		bool switch_left_right = false;
		if (large_solution_set->Size() < small_solution_set->Size()) {
			std::swap(small_solution_set, large_solution_set);
			switch_left_right = true;
		}
		if (!reconstruct && (left_solutions->GetPruned() || right_solutions->GetPruned())) final_solutions->SetPruned();
		if (!reconstruct && (!IMB_SORT || max_depth <= 2 || current_depth > 0)) {
			//Filter based on upper bound
			switch_left_right = !switch_left_right;
			bool large_sorted = false;
			if (upper_bound < INT32_MAX) {
				if (true || std::log(double(large_solution_set->Size())) < small_solution_set->Size()) {
					large_solution_set->SortByMisclassifications();
					large_sorted = true;
				}
				small_solution_set->SortByMisclassifications();
			}
			for (auto& out_sol : small_solution_set->solutions) {
				if (out_sol.GetMisclassifications() + large_solution_set->GetLeastMisclassifications() >= upper_bound) break;
				if (reconstruct && large_sorted && out_sol.GetMisclassifications() + large_solution_set->solutions[large_solution_set->Size()-1].GetMisclassifications() < tree_node->parent.GetMisclassifications()) continue;
				for (auto& in_sol : large_solution_set->solutions) {
					auto& left_sol = switch_left_right ? in_sol : out_sol;
					auto& right_sol = switch_left_right ? out_sol : in_sol;
					if (reconstruct) {
						if (CheckSolution(left_sol, right_sol, tree_node)) return;
					} else {
						MergeAdd(feature, branch, upper_bound, left_sol, right_sol, final_solutions);
					}
					if (large_sorted && left_sol.GetMisclassifications() + right_sol.GetMisclassifications() >= upper_bound) continue;
					i++;
					if (DEBUG_PRINT && current_depth <= 1 && i % delta == 0) {
						for (int i = 0; i < current_depth; i++) std::cout << "  ";
						std::cout << "Merged " << (i / delta) << "0M/" << (total / delta) << "0M combinations so far (" << (i * 100 / total) << "%)";
						if(!reconstruct)
							std::cout << ", resulting in " << final_solutions->Size() << " solutions "
							<< "UB = " << upper_bound << ".";
						std::cout << std::endl;
					}
				}
				if (current_depth == 0 && !stopwatch_.IsWithinTimeLimit()) return;
			}
		} else { // If current_depth == 0, or if reconstructing, filter based on fairness
			auto outer_container = large_solution_set;
			auto inner_container = small_solution_set;
			inner_container->SortByInbalance();
			int inner_lower_bound = 0;
			if (upper_bound < INT32_MAX) {
				inner_lower_bound = inner_container->GetLeastMisclassifications();
				outer_container->SortByMisclassifications();
			}

			
			for (auto& out_sol : outer_container->solutions) {
				if (out_sol.GetMisclassifications() + inner_lower_bound >= upper_bound) break;
				auto& left_sol_p = out_sol.GetPartialSolution();
				double disc_lower_bound = -discrimination_cutoff_value - out_sol.GetPartialSolution().GetInbalance(data_summary);
				double disc_upper_bound = discrimination_cutoff_value - out_sol.GetPartialSolution().GetInbalance(data_summary);
				if (reconstruct) {
					disc_lower_bound = -DISC_EPS + tree_node->parent.node_compare.partial_discrimination - out_sol.GetPartialSolution().GetInbalance(data_summary);
					disc_upper_bound = DISC_EPS + tree_node->parent.node_compare.partial_discrimination - out_sol.GetPartialSolution().GetInbalance(data_summary);
				}
				runtime_assert(disc_lower_bound <= disc_upper_bound + DBL_EPSILON);
				size_t ix = inner_container->LowerBoundByInbalance(data_summary, disc_lower_bound);
				size_t end = inner_container->UpperBoundByInbalance(data_summary, disc_upper_bound);
				stats.num_partial_solution_candidates_relaxed_fair[current_depth] += end - ix;
				for (; ix < end; ix++) {
					auto& in_sol = inner_container->solutions[ix];
					auto& left_sol = switch_left_right ? in_sol : out_sol;
					auto& right_sol = switch_left_right ? out_sol : in_sol;
					if (reconstruct) {
						if (CheckSolution(left_sol, right_sol, tree_node)) return;
					} else {
						MergeAdd(feature, branch, upper_bound, left_sol, right_sol, final_solutions);
					}
					i++;
					if (DEBUG_PRINT && current_depth <= 1 && i % delta == 0) {
						for (int i = 0; i < current_depth; i++) std::cout << "  ";
						std::cout << "Merged " << (i / delta) << "0M/" << (total / delta) << "0M combinations so far (" << (i * 100 / total) << "%)";
						if (!reconstruct)
							std::cout << ", resulting in " << final_solutions->Size() << " solutions "
								<< "UB = " << upper_bound << ".";
						std::cout << std::endl;
					}
				}
				if (current_depth == 0 && !stopwatch_.IsWithinTimeLimit()) return;
			}
		}
		runtime_assert(!reconstruct)
		//if (!return_pareto_front && upper_bound != INT32_MAX && upper_bound + 1 < initial_upper_bound) {
		//	final_solutions->FilterOnUpperbound(upper_bound + 1); // Plus one in order not to remove the optimal solution
		//}
		double time_inc = double(clock() - clock_start) / CLOCKS_PER_SEC;
		stats.time_merging += time_inc;
		stats.time_merging_per_layer[current_depth] += time_inc;
	}

#else

	template<bool reconstruct>
	void Solver::Merge(int feature, const Branch& branch, const Branch& left_branch_, const Branch& right_branch_,
		int& upper_bound, AssignmentContainer* left_solutions, AssignmentContainer* right_solutions, AssignmentContainer* final_solutions, InternalTreeNode* tree_node) {
		if (left_solutions->Size() == 0 || right_solutions->Size() == 0) return;
		runtime_assert(reconstruct != (tree_node == nullptr));
		clock_t clock_start = clock();	
		const int initial_upper_bound = upper_bound;
		const int current_depth = branch.Depth();
		size_t total = left_solutions->Size() * right_solutions->Size();
		stats.num_partial_solution_candidates[current_depth] += total;

		auto large_solution_set = left_solutions;
		auto small_solution_set = right_solutions;
		bool switch_left_right = false;

		if (!reconstruct && (left_solutions->GetPruned() || right_solutions->GetPruned())) final_solutions->SetPruned();
		{
			switch_left_right = !switch_left_right;
			for (auto& out_sol : small_solution_set->solutions) {
				for (auto& in_sol : large_solution_set->solutions) {
					auto& left_sol = switch_left_right ? in_sol : out_sol;
					auto& right_sol = switch_left_right ? out_sol : in_sol;
					if (reconstruct) {
						if (CheckSolution(left_sol, right_sol, tree_node)) return;
					} else {
						MergeAdd(feature, branch, upper_bound, left_sol, right_sol, final_solutions);
					}
				}
				if (current_depth == 0 && !stopwatch_.IsWithinTimeLimit()) return;
			}
		}
		runtime_assert(!reconstruct)
		//if (!return_pareto_front && upper_bound != INT32_MAX && upper_bound + 1 < initial_upper_bound) {
		//	final_solutions->FilterOnUpperbound(upper_bound + 1); // Plus one in order not to remove the optimal solution
		//}
		double time_inc = double(clock() - clock_start) / CLOCKS_PER_SEC;
		stats.time_merging += time_inc;
		stats.time_merging_per_layer[current_depth] += time_inc;
	}

#endif

	void Solver::MergeAdd(int feature, const Branch& branch, int& upper_bound, const InternalNodeDescription& left_sol, const InternalNodeDescription& right_sol, AssignmentContainer* final_solutions) {
		if (left_sol.GetMisclassifications() + right_sol.GetMisclassifications() >= upper_bound) return;
		const int current_depth = branch.Depth();
		stats.num_partial_solution_candidates_within_upperbound[current_depth]++;
		auto new_sol_part = PartialSolution::Merge(left_sol.GetPartialSolution(), right_sol.GetPartialSolution());
		int num_nodes = 1 + left_sol.NumNodes() + right_sol.NumNodes();
		if (final_solutions->Contains(new_sol_part, num_nodes, data_summary)) return;
		stats.num_partial_solution_candidates_unique[current_depth]++;
		auto new_sol = InternalNodeDescription(feature, INT32_MAX, new_sol_part,
			left_sol.NumNodes(), right_sol.NumNodes(), branch, data_summary);
		if (!USE_PRUNE || new_sol.GetBestDiscrimination() <= discrimination_cutoff_value) {
			stats.num_partial_solution_candidates_possibly_fair[current_depth]++;
			final_solutions->Add(new_sol, false);
			if (final_solutions->Contains(new_sol_part, num_nodes, data_summary)) stats.num_computed_non_dom_nodes[current_depth]++;
#if USE_PRUNE
			if(new_sol.GetMisclassifications() < upper_bound && ((return_pareto_front && new_sol.GetWorstDiscrimination() <= DISC_EPS) ||
			   (!return_pareto_front &&  new_sol.GetWorstDiscrimination() <= discrimination_cutoff_value))) { 
				upper_bound = new_sol.GetMisclassifications();
				final_solutions->SetPruned();
			}
#else
			if (current_depth == 0 && new_sol.GetMisclassifications() < upper_bound && ((return_pareto_front && new_sol.GetWorstDiscrimination() <= DISC_EPS) ||
				(!return_pareto_front && new_sol.GetWorstDiscrimination() <= discrimination_cutoff_value))) {
				upper_bound = new_sol.GetMisclassifications();
				final_solutions->SetPruned();
			}
#endif
		} else {
			final_solutions->SetPruned();
		}
	}



	bool Solver::CheckSolution(const InternalNodeDescription& n1, const InternalNodeDescription& n2, InternalTreeNode* tree_node) {
		const auto& sol = tree_node->parent;
		if (DEBUG_PRINT) {
			std::cout << "Search for node ("
				<< sol.num_nodes_left << ", "
				<< sol.num_nodes_right << ", "
				<< sol.GetMisclassifications() << ", "
				<< sol.node_compare.partial_discrimination << ")\t";
			std::cout << " Check ("
				<< n1.NumNodes() << ", "
				<< n2.NumNodes() << ", "
				<< n1.GetMisclassifications() + n2.GetMisclassifications() << ", "
				<< n1.node_compare.partial_discrimination + n2.node_compare.partial_discrimination << ")" << std::endl;
		}
		if (n1.GetMisclassifications() + n2.GetMisclassifications() != sol.GetMisclassifications() ||
			std::abs(n1.node_compare.partial_discrimination + n2.node_compare.partial_discrimination - sol.node_compare.partial_discrimination) >= DISC_EPS
			) return false;
		tree_node->left_child = n1;
		tree_node->right_child = n2;
		return true;
	}

	bool Solver::UpdateCacheUsingSimilarity(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound) {
		// Compute the similarity-based lower bound (Section 4.5) and update current bound
		PairLowerBoundOptimal result = similarity_lower_bound_computer->ComputeLowerBound(data, branch, max_depth, num_nodes, upper_bound, cache);
		if (result.optimal) { return true; }
		if (result.lower_bound > 0) { cache->UpdateLowerBound(data, branch, result.lower_bound, max_depth, num_nodes); }
		return false;
	}

	bool Solver::SatisfiesConstraint(const PartialSolution& solution, const Branch& branch) const {
		return solution.GetBestDiscrimination(branch, data_summary) <= discrimination_cutoff_value + DISC_EPS;
	}

	std::shared_ptr<AssignmentContainer> Solver::CreateLeafNodeDescriptions(const BinaryData& data, const Branch& branch, int accuracy_upper_bound) const {
		auto container = std::make_shared<AssignmentContainer>(branch.Depth() == 0, 1);
		if (data.Size() < minimum_leaf_node_size) return container;
		auto sol1 = InternalNodeDescription(INT32_MAX, 0, CreatePartialSolution(data, 0), 0, 0, branch, data_summary);
		auto sol2 = InternalNodeDescription(INT32_MAX, 1, CreatePartialSolution(data, 1), 0, 0, branch, data_summary);
		if (sol1.GetMisclassifications() < accuracy_upper_bound) {
			if (!USE_PRUNE || SatisfiesConstraint(sol1)) container->Add(sol1);
			else container->SetPruned();
		}
		if (sol2.GetMisclassifications() < accuracy_upper_bound) {
			if (!USE_PRUNE || SatisfiesConstraint(sol2)) container->Add(sol2);
			else container->SetPruned();
		}
		container->Filter(data_summary);
		return container;
	}

	PartialSolution Solver::CreatePartialSolution(const BinaryData& data, int label) const {
		int misclassifications = data.NumInstancesForLabel(1 - label);
		int group0_pos = label ? data.NumInstancesForGroup(0) : 0;
		int group1_pos = label ? data.NumInstancesForGroup(1) : 0;
		return PartialSolution(misclassifications, group0_pos, group1_pos);
	}

	DiscriminationBudget Solver::GetDiscriminationBudget(const BinaryData& other_data, const Branch& branch) const {
#if USE_PRUNE
		auto& budget = branch.GetDiscriminationBudget();
		double other_min = -((double)other_data.NumInstancesForGroup(1)) / data_summary.group1_size;
		double other_max = ((double)other_data.NumInstancesForGroup(0)) / data_summary.group0_size;;
		return { budget.min_balance + other_min,
				budget.max_balance + other_max };
#else
		return { -1.0, 1.0 };
#endif
	}

	void Solver::UpdateDiscriminationBudget(const Branch& org_branch, Branch& sub_branch, AssignmentContainer* partial_solutions) const {
#if USE_PRUNE
		auto& org_budget = org_branch.GetDiscriminationBudget();
		auto& sub_budget = sub_branch.GetDiscriminationBudget();
		double min_balance = std::min(org_budget.min_balance + partial_solutions->GetMinBalance(data_summary), sub_budget.min_balance);
		double max_balance = std::min(org_budget.max_balance + partial_solutions->GetMaxBalance(data_summary), sub_budget.max_balance);
		sub_branch.SetDiscriminationBudget({ min_balance, max_balance });
#endif
	}

	DiscriminationBudget Solver::GetDiscriminationBudget(int group0, int group1, const Branch& branch) const {
#if USE_PRUNE
		auto& budget = branch.GetDiscriminationBudget();
		double other_min = -double(group1) / data_summary.group1_size;
		double other_max = double(group0) / data_summary.group0_size;
		return { budget.min_balance + other_min,
				budget.max_balance + other_max
		};
#else
		return { -1.0, 1.0 };
#endif
	}

	std::shared_ptr<DecisionNode> Solver::ConstructOptimalTree(const InternalNodeDescription& node, BinaryData& data, Branch& branch, int max_depth, int num_nodes) {
		runtime_assert(num_nodes >= 0);
		//Update branch based on internalnodedescription
		
		int branch_lower_bound = cache->RetrieveLowerBound(data, branch, max_depth, num_nodes);
		if (branch_lower_bound > 0) stats.num_cache_hit_nonzero_bound++;
		if (max_depth == 0 || num_nodes == 0 ) {
			return DecisionNode::CreateLabelNode(node.label);
		} 
		auto tree = DecisionNode::CreateFeatureNodeWithNullChildren(node.feature);
		int upper_bound = node.GetMisclassifications() + 1;

		BinaryData left_data;
		BinaryData right_data;
		data.SplitData(node.feature, left_data, right_data);
		runtime_assert(left_data.Size() >= minimum_leaf_node_size && right_data.Size() >= minimum_leaf_node_size);

		BinaryData* large_data = &left_data;
		BinaryData* small_data = &right_data;

		Branch left_branch = Branch::LeftChildBranch(branch, node.feature, GetDiscriminationBudget(*small_data, branch));
		Branch right_branch = Branch::RightChildBranch(branch, node.feature, GetDiscriminationBudget(*large_data, branch));
		

		int left_subtree_size = node.num_nodes_left;
		int right_subtree_size = node.num_nodes_right;
		int left_depth = std::min(max_depth - 1, left_subtree_size);
		int right_depth = std::min(max_depth - 1, right_subtree_size);

		bool switch_left_right = false;
		if (small_data->Size() < large_data->Size()) {
			std::swap(small_data, large_data);
			std::swap(left_branch, right_branch);
			std::swap(left_subtree_size, right_subtree_size);
			std::swap(left_depth, right_depth);
			switch_left_right = true;
		}


		int right_lower_bound = cache->RetrieveLowerBound(*small_data, right_branch, right_depth, right_subtree_size);
		if (right_lower_bound > 0) stats.num_cache_hit_nonzero_bound++;
		int left_upper_bound = upper_bound - right_lower_bound;
		int left_lower_bound = cache->RetrieveLowerBound(*large_data, left_branch, left_depth, left_subtree_size);
		if (left_lower_bound > 0) stats.num_cache_hit_nonzero_bound++;
		int right_upper_bound = upper_bound - left_lower_bound;
		if(USE_PRUNE) cache->UpdateDiscrimationBudget(branch, left_branch, *small_data, right_branch, right_depth, right_subtree_size, right_upper_bound);
		auto left_solutions = SolveSubtree(*large_data, left_branch, left_depth, left_subtree_size, left_upper_bound);
		runtime_assert(left_solutions->Size() > 0);
		right_upper_bound = upper_bound - left_solutions->GetLeastMisclassifications();
		if (USE_PRUNE) UpdateDiscriminationBudget(branch, right_branch, left_solutions.get());
		auto right_solutions = SolveSubtree(*small_data, right_branch, right_depth, right_subtree_size, right_upper_bound);
		runtime_assert(right_solutions->Size() > 0);

		if (switch_left_right) {
			std::swap(small_data, large_data);
			std::swap(left_branch, right_branch);
			std::swap(left_solutions, right_solutions);
			std::swap(left_subtree_size, right_subtree_size);
			std::swap(left_depth, right_depth);
		}

		//Merge solutions and find the current node
		left_solutions->FilterOnNumberOfNodes(node.num_nodes_left);
		right_solutions->FilterOnNumberOfNodes(node.num_nodes_right);

		while(true) {
			size_t size_left = left_solutions->Size();
			size_t size_right = right_solutions->Size();
			const double sol_disc = node.node_compare.partial_discrimination;
			right_solutions->UpdateBestBounds(data_summary);
			left_solutions->FilterOnImbalance(sol_disc - right_solutions->GetMaxBalance(data_summary), sol_disc - right_solutions->GetMinBalance(data_summary), data_summary);
			left_solutions->UpdateBestBounds(data_summary);
			right_solutions->FilterOnImbalance(sol_disc - left_solutions->GetMaxBalance(data_summary), sol_disc - left_solutions->GetMinBalance(data_summary), data_summary);
			if (left_solutions->Size() == size_left && right_solutions->Size() == size_right) break;
		}
		
		InternalTreeNode tree_node;
		tree_node.parent = node;

		Merge<true>(node.feature, branch, left_branch, right_branch, upper_bound, left_solutions.get(), right_solutions.get(), nullptr, &tree_node);
		runtime_assert(tree_node.left_child.IsFeasible());
		runtime_assert(tree_node.right_child.IsFeasible());

		tree->left_child = ConstructOptimalTree(tree_node.left_child, *large_data, left_branch, left_depth, left_subtree_size);
		tree->right_child = ConstructOptimalTree(tree_node.right_child, *small_data, right_branch, right_depth, right_subtree_size);
		return tree;
	}
}