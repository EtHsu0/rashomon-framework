/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/terminal_solver.h"
#include "solver/solver.h"

namespace DPF {

	TerminalSolver::TerminalSolver(Solver* solver, int num_features) :
		solver(solver),
		num_features(num_features),
		frequency_counter(num_features),
		best_children_info(num_features),
		left_branches(num_features),
		right_branches(num_features)
	{
		
	}

	TerminalResults TerminalSolver::Solve(const BinaryData& data, const Branch& branch, int num_nodes, int upper_bound) {
		results.Clear();
		if (num_nodes == 1) {
			results.one_node_solutions =  SolveOneNode(data, branch, upper_bound, false); 
			return results;
		}
		InitialiseChildrenInfo();
		frequency_counter.Initialize(data);
		InitializeBranches(branch, data);
		
		// Possibly include code to reuse previous calculations

		results.one_node_solutions = SolveOneNode(data, branch, upper_bound, true);
		
		for (int f1 = 0; f1 < num_features; f1++) {
			for (int f2 = 0; f2 < num_features; f2++) {
				if (f1 == f2) continue;

				int total00 = frequency_counter.NegativesZeroZero(f1, f2) + frequency_counter.PositivesZeroZero(f1, f2);
				PartialSolution sol000, sol001, sol010, sol011, sol100, sol101, sol110, sol111;
				if (total00 >= solver->GetMinimumLeafNodeSize()) {
					sol000 = PartialSolution(frequency_counter.PositivesZeroZero(f1, f2), 0, 0);
					sol001 = PartialSolution(frequency_counter.NegativesZeroZero(f1, f2),
						frequency_counter.GroupZeroZero(0, f1, f2), frequency_counter.GroupZeroZero(1, f1, f2));
				}

				int total01 = frequency_counter.NegativesZeroOne(f1, f2) + frequency_counter.PositivesZeroOne(f1, f2);
				if (total01 >= solver->GetMinimumLeafNodeSize()) {
					sol010 = PartialSolution(frequency_counter.PositivesZeroOne(f1, f2), 0, 0);
					sol011 = PartialSolution(frequency_counter.NegativesZeroOne(f1, f2),
						frequency_counter.GroupZeroOne(0, f1, f2), frequency_counter.GroupZeroOne(1, f1, f2));
				}

				int total10 = frequency_counter.NegativesOneZero(f1, f2) + frequency_counter.PositivesOneZero(f1, f2);
				if (total10 >= solver->GetMinimumLeafNodeSize()) {
					sol100 = PartialSolution(frequency_counter.PositivesOneZero(f1, f2), 0, 0);
					sol101 = PartialSolution(frequency_counter.NegativesOneZero(f1, f2),
						frequency_counter.GroupOneZero(0, f1, f2), frequency_counter.GroupOneZero(1, f1, f2));
				}

				int total11 = frequency_counter.NegativesOneOne(f1, f2) + frequency_counter.PositivesOneOne(f1, f2);
				if (total11 >= solver->GetMinimumLeafNodeSize()) {
					sol110 = PartialSolution(frequency_counter.PositivesOneOne(f1, f2), 0, 0);
					sol111 = PartialSolution(frequency_counter.NegativesOneOne(f1, f2),
						frequency_counter.GroupOneOne(0, f1, f2), frequency_counter.GroupOneOne(1, f1, f2));
				}

				// Find best left child (first=0)
				if (total00 >= solver->GetMinimumLeafNodeSize() && total01 >= solver->GetMinimumLeafNodeSize()) {
					PartialSolution left_sol = PartialSolution::Merge(sol000, sol011); // label (0,1)
					UpdateBestLeftChild(f1, left_sol, upper_bound);
					left_sol = PartialSolution::Merge(sol001, sol010); // label (1,0)
					UpdateBestLeftChild(f1, left_sol, upper_bound);
				}
				// Find best right child (first=1)
				if (total10 >= solver->GetMinimumLeafNodeSize() && total11 >= solver->GetMinimumLeafNodeSize()) {
					PartialSolution right_sol = PartialSolution::Merge(sol100, sol111); // label (0,1)
					UpdateBestRightChild(f1, right_sol, upper_bound);
					right_sol = PartialSolution::Merge(sol101, sol110); // label (1,0)
					UpdateBestRightChild(f1, right_sol, upper_bound);
				}
			}
			if(num_nodes == 3) UpdateBestThreeNodeAssignment(branch, f1, upper_bound);
			UpdateBestTwoNodeAssignment(branch, f1, upper_bound);
		}
		results.two_nodes_solutions->Add(results.one_node_solutions.get());
		if (num_nodes == 2) {
			results.UpdateBestBounds(solver->GetDataSummary());
			return results;
		} else { 
			runtime_assert(num_nodes == 3);
			results.three_nodes_solutions->Add(results.two_nodes_solutions.get());
			results.UpdateBestBounds(solver->GetDataSummary());
			return results;
		}
	}

	void TerminalSolver::UpdateBestTwoNodeAssignment(const Branch& branch, int root_feature, int upper_bound) {

		AssignmentContainer left_leaves(false, 0), right_leaves(false, 0);

		int left_size = frequency_counter.PositivesZeroZero(root_feature, root_feature) + frequency_counter.NegativesZeroZero(root_feature, root_feature);
		if (left_size >= solver->GetMinimumLeafNodeSize()) {
			PartialSolution left_leaf_label0 (frequency_counter.PositivesZeroZero(root_feature, root_feature), 0, 0);
			PartialSolution left_leaf_label1 (frequency_counter.NegativesZeroZero(root_feature, root_feature),
				frequency_counter.GroupZeroZero(0, root_feature, root_feature), frequency_counter.GroupZeroZero(1, root_feature, root_feature));
			InternalNodeDescription left_leaf_sol_label0(INT32_MAX, 0, left_leaf_label0, 0, 0, left_branches[root_feature], solver->GetDataSummary());
			InternalNodeDescription left_leaf_sol_label1(INT32_MAX, 1, left_leaf_label1, 0, 0, left_branches[root_feature], solver->GetDataSummary());
			if (left_leaf_sol_label0.GetMisclassifications() < upper_bound) { 
				if (!USE_PRUNE || solver->SatisfiesConstraint(left_leaf_sol_label0)) left_leaves.Add(left_leaf_sol_label0);
				else left_leaves.SetPruned();
			}
			if (left_leaf_sol_label1.GetMisclassifications() < upper_bound) {
				if (!USE_PRUNE || solver->SatisfiesConstraint(left_leaf_sol_label1)) left_leaves.Add(left_leaf_sol_label1);
				else left_leaves.SetPruned();
			}
		}

		
		int right_size = frequency_counter.PositivesOneOne(root_feature, root_feature) + frequency_counter.NegativesOneOne(root_feature, root_feature);
		if (right_size >= solver->GetMinimumLeafNodeSize()) {
			PartialSolution right_leaf_label0 (frequency_counter.PositivesOneOne(root_feature, root_feature), 0, 0);
			PartialSolution right_leaf_label1 (frequency_counter.NegativesOneOne(root_feature, root_feature),
				frequency_counter.GroupOneOne(0, root_feature, root_feature), frequency_counter.GroupOneOne(1, root_feature, root_feature));
			InternalNodeDescription right_leaf_sol_label0(INT32_MAX, 0, right_leaf_label0, 0, 0, right_branches[root_feature], solver->GetDataSummary());
			InternalNodeDescription right_leaf_sol_label1(INT32_MAX, 1, right_leaf_label1, 0, 0, right_branches[root_feature], solver->GetDataSummary());
			if (right_leaf_sol_label0.GetMisclassifications() < upper_bound) {
				if (!USE_PRUNE || solver->SatisfiesConstraint(right_leaf_sol_label0)) right_leaves.Add(right_leaf_sol_label0);
				else right_leaves.SetPruned();
			}
			if (right_leaf_sol_label1.GetMisclassifications() < upper_bound) {
				if (!USE_PRUNE || solver->SatisfiesConstraint(right_leaf_sol_label1)) right_leaves.Add(right_leaf_sol_label1);
				else right_leaves.SetPruned();
			}
		}

		auto left_children = best_children_info[root_feature].left_child_assignments;
		auto right_children = best_children_info[root_feature].right_child_assignments;
		
		if (right_leaves.Size() > 0) {
			
			Merge(root_feature, branch, left_branches[root_feature], right_branches[root_feature],
				upper_bound, left_children.get(), &right_leaves, results.two_nodes_solutions.get());
		}
		if (left_leaves.Size() > 0) {
			Merge(root_feature, branch, left_branches[root_feature], right_branches[root_feature],
				upper_bound, &left_leaves, right_children.get(), results.two_nodes_solutions.get());
		}
	}

	void TerminalSolver::UpdateBestThreeNodeAssignment(const Branch& branch, int root_feature, int upper_bound) {
		auto left_children = best_children_info[root_feature].left_child_assignments;
		auto right_children = best_children_info[root_feature].right_child_assignments;
		Merge(root_feature, branch, left_branches[root_feature], right_branches[root_feature],
			upper_bound, left_children.get(), right_children.get(), results.three_nodes_solutions.get());
	}

	void TerminalSolver::UpdateBestLeftChild(int root_feature, const PartialSolution& solution, int upper_bound) {
		InternalNodeDescription sol(root_feature, INT32_MAX, solution, 0, 0, left_branches[root_feature], solver->GetDataSummary());
		if (sol.GetMisclassifications() < upper_bound) {
			if (!USE_PRUNE || solver->SatisfiesConstraint(sol)) best_children_info[root_feature].left_child_assignments->Add(sol);
			else best_children_info[root_feature].left_child_assignments->SetPruned();
		}
	}

	void TerminalSolver::UpdateBestRightChild(int root_feature, const PartialSolution& solution, int upper_bound) {
		InternalNodeDescription sol(root_feature, INT32_MAX, solution, 0, 0, right_branches[root_feature], solver->GetDataSummary());
		if (sol.GetMisclassifications() < upper_bound) {
			if (!USE_PRUNE || solver->SatisfiesConstraint(sol)) best_children_info[root_feature].right_child_assignments->Add(sol);
			else best_children_info[root_feature].right_child_assignments->SetPruned();
		}
	}

	void TerminalSolver::Merge(int feature, const Branch& branch, const Branch& left_branch, const Branch& right_branch, int& upper_bound,
		AssignmentContainer* left_solutions, AssignmentContainer* right_solutions, AssignmentContainer* final_solutions) {
		if (left_solutions->Size() == 0 || right_solutions->Size() == 0) return;
		if (left_solutions->GetPruned() || right_solutions->GetPruned()) final_solutions->SetPruned();
		{
			solver->GetStatistics().num_partial_solution_candidates[branch.Depth()] += left_solutions->Size() * right_solutions->Size();
			for (auto& left_sol : left_solutions->solutions) {
				for (auto& right_sol : right_solutions->solutions) {
					solver->MergeAdd(feature, branch, upper_bound, left_sol, right_sol, final_solutions);
				}
			}
		} 
	}

	std::shared_ptr<AssignmentContainer> TerminalSolver::SolveOneNode(const BinaryData& data, const Branch& branch, int upper_bound, bool initialized) {
		std::shared_ptr<AssignmentContainer> solutions = solver->CreateLeafNodeDescriptions(data, branch, upper_bound);
		
		int* positives_left = new int[num_features];
		int* positives_right = new int[num_features];
		int* negatives_left = new int[num_features];
		int* negatives_right = new int[num_features];
		int* group0_left = new int[num_features];
		int* group1_left = new int[num_features];
		int* group0_right = new int[num_features];
		int* group1_right = new int[num_features];

		if (initialized) {
			for (int feature = 0; feature < num_features; feature++) {
				positives_left[feature]  = frequency_counter.PositivesZeroZero(feature, feature);
				positives_right[feature] = frequency_counter.PositivesOneOne(feature, feature);
				negatives_left[feature]	 = frequency_counter.NegativesZeroZero(feature, feature);
				negatives_right[feature] = frequency_counter.NegativesOneOne(feature, feature);
				group0_left[feature]	 = frequency_counter.GroupZeroZero(0, feature, feature);
				group1_left[feature]	 = frequency_counter.GroupZeroZero(1, feature, feature);
				group0_right[feature]	 = frequency_counter.GroupOneOne(0, feature, feature);
				group1_right[feature]	 = frequency_counter.GroupOneOne(1, feature, feature);
			}
		} else {
			for (int feature = 0; feature < num_features; feature++) {
				positives_left[feature] = 0;
				positives_right[feature] = 0;
				negatives_left[feature] = 0;
				negatives_right[feature] = 0;
				group0_left[feature] = 0;
				group1_left[feature] = 0;
				group0_right[feature] = 0;
				group1_right[feature] = 0;
			}
			for (int d_ix = 0; d_ix < data.Size(); d_ix++) {
				const auto data_point = data.GetInstance(d_ix);
				int group = data.GetGroup(d_ix);
				int label = data.GetLabel(d_ix);
				for (int feature = 0; feature < num_features; feature++) {
					if (!data_point->IsFeaturePresent(feature)) {
						if (label)  positives_left[feature]++;
						else		negatives_left[feature]++;
						if (group)  group1_left[feature]++;
						else		group0_left[feature]++;
					} else {
						if (label)  positives_right[feature]++;
						else		negatives_right[feature]++;
						if (group)  group1_right[feature]++;
						else		group0_right[feature]++;
					}
				}
			}
		}

		for (int feature = 0; feature < num_features; feature++) {
			int total_left = positives_left[feature] + negatives_left[feature];
			int total_right = positives_right[feature] + negatives_right[feature];
			if (total_left < solver->GetMinimumLeafNodeSize() || total_right < solver->GetMinimumLeafNodeSize()) continue;

			PartialSolution left_label0(positives_left[feature], 0, 0);
			PartialSolution left_label1(negatives_left[feature], 
				group0_left[feature], group1_left[feature]);
			
			PartialSolution right_label0(positives_right[feature], 0, 0);
			PartialSolution right_label1(negatives_right[feature],
				group0_right[feature], group1_right[feature]);

			
			auto left0_right1 = PartialSolution::Merge(left_label0, right_label1);
			auto left1_right0 = PartialSolution::Merge(left_label1, right_label0);
			
			InternalNodeDescription sol01(feature, INT32_MAX, left0_right1, 0, 0, branch, solver->GetDataSummary());
			InternalNodeDescription sol10(feature, INT32_MAX, left1_right0, 0, 0, branch, solver->GetDataSummary());

			if (sol01.GetMisclassifications() < upper_bound) {
				if (!USE_PRUNE || solver->SatisfiesConstraint(sol01)) solutions->Add(sol01);
				else solutions->SetPruned();
			}
			if (sol10.GetMisclassifications() < upper_bound) {
				if (!USE_PRUNE || solver->SatisfiesConstraint(sol10)) solutions->Add(sol10);
				else solutions->SetPruned();
			}
			
		}

		delete[] positives_left;
		delete[] positives_right;
		delete[] negatives_left;
		delete[] negatives_right;
		delete[] group0_left;
		delete[]  group1_left;
		delete[] group0_right;
		delete[]  group1_right;
		solutions->UpdateBestBounds(solver->GetDataSummary());
		return solutions;
	}

	void TerminalSolver::InitialiseChildrenInfo() {
		for (int i = 0; i < num_features; i++) {
			best_children_info[i].Clear();
		}
	}

	void TerminalSolver::InitializeBranches(const Branch& branch, const BinaryData& data) {
		for (int f = 0; f < num_features; f++) {
			int group0_left = frequency_counter.GroupZeroZero(0, f, f);
			int group0_right = frequency_counter.GroupOneOne(0, f, f);
			int group1_left = frequency_counter.GroupZeroZero(1, f, f);
			int group1_right = frequency_counter.GroupOneOne(1, f, f);
			left_branches[f] = Branch::LeftChildBranch(branch, f, solver->GetDiscriminationBudget(group0_right, group1_right, branch));
			right_branches[f] = Branch::RightChildBranch(branch, f, solver->GetDiscriminationBudget(group0_left, group1_left, branch));
		}
	}
}