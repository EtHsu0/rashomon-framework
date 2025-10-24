/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once

const int max_depth = 10;

namespace DPF {

	struct Statistics {
		Statistics() {
			num_terminal_nodes_with_node_budget_one = 0;
			num_terminal_nodes_with_node_budget_two = 0;
			num_terminal_nodes_with_node_budget_three = 0;
			std::fill(num_computed_nodes, num_computed_nodes + max_depth, 0);
			std::fill(num_computed_non_dom_nodes, num_computed_non_dom_nodes + max_depth, 0);
			std::fill(num_partial_solution_candidates, num_partial_solution_candidates + max_depth, 0);
			std::fill(num_partial_solution_candidates_relaxed_fair, num_partial_solution_candidates_relaxed_fair + max_depth, 0);
			std::fill(num_partial_solution_candidates_within_upperbound, num_partial_solution_candidates_within_upperbound + max_depth, 0);
			std::fill(num_partial_solution_candidates_unique, num_partial_solution_candidates_unique + max_depth, 0);
			std::fill(num_partial_solution_candidates_possibly_fair, num_partial_solution_candidates_possibly_fair + max_depth, 0);			
			std::fill(time_merging_per_layer, time_merging_per_layer + max_depth, 0.0);

			time_in_terminal_node = 0;
			time_merging = 0;

			num_cache_hit_nonzero_bound = 0;
			num_cache_hit_optimality = 0;
		}
		size_t num_terminal_nodes_with_node_budget_one;
		size_t num_terminal_nodes_with_node_budget_two;
		size_t num_terminal_nodes_with_node_budget_three;
		size_t num_computed_nodes[max_depth];
		size_t num_computed_non_dom_nodes[max_depth];
		size_t num_partial_solution_candidates[max_depth];
		size_t num_partial_solution_candidates_relaxed_fair[max_depth];
		size_t num_partial_solution_candidates_within_upperbound[max_depth];
		size_t num_partial_solution_candidates_unique[max_depth];
		size_t num_partial_solution_candidates_possibly_fair[max_depth];

		size_t num_cache_hit_optimality;
		size_t num_cache_hit_nonzero_bound;


		double time_in_terminal_node;
		double time_merging;
		double time_merging_per_layer[max_depth];
	};
}