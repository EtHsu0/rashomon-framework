/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "model/container.h"
#include "utils/parameter_handler.h"
#include "solver/counter.h"
#include "solver/difference_computer.h"

namespace DPF {

	class FrequencyCounter {
	public:
		FrequencyCounter(int num_features);
		
		void Initialize(const BinaryData& data);
		int ProbeDifference(const BinaryData& data) const;

		int PositivesZeroZero(int f1, int f2) const;
		int PositivesZeroOne(int f1, int f2) const;
		int PositivesOneZero(int f1, int f2) const;
		int PositivesOneOne(int f1, int f2) const;

		int NegativesZeroZero(int f1, int f2) const;
		int NegativesZeroOne(int f1, int f2) const;
		int NegativesOneZero(int f1, int f2) const;
		int NegativesOneOne(int f1, int f2) const;

		int GroupZeroZero(int a, int f1, int f2) const;
		int GroupZeroOne(int a, int f1, int f2) const;
		int GroupOneZero(int a, int f1, int f2) const;
		int GroupOneOne(int a, int f1, int f2) const;

		void UpdateCounts(const BinaryData& data, int value);

	private:
		BinaryData data;
		Counter counts_group0;
		Counter counts_group1;
	};

	
	struct TerminalResults {
		TerminalResults() { Clear(); }
		void Clear() {
			one_node_solutions = std::make_shared<AssignmentContainer>(false, 1);
			two_nodes_solutions = std::make_shared<AssignmentContainer>(false, 2);
			three_nodes_solutions = std::make_shared<AssignmentContainer>(false, 3);
		}
		void UpdateBestBounds(const DataSummary& data_summary) {
			one_node_solutions->UpdateBestBounds(data_summary);
			two_nodes_solutions->UpdateBestBounds(data_summary);
			three_nodes_solutions->UpdateBestBounds(data_summary);
		}
		std::shared_ptr<AssignmentContainer> one_node_solutions, two_nodes_solutions, three_nodes_solutions;
	};
	
	class Solver;

	class TerminalSolver {
	public:
		TerminalSolver(Solver* solver, int num_features);
		TerminalResults Solve(const BinaryData& data, const Branch& branch, int num_nodes, int upper_bound);
		inline int ProbeDifference(const BinaryData& data) const { return frequency_counter.ProbeDifference(data); }
		inline TerminalResults GetResults() const { return results; }

	private:

		std::shared_ptr<AssignmentContainer> SolveOneNode(const BinaryData& data, const Branch& branch, int upper_bound, bool initialized);

		struct ChildrenInformation {
			ChildrenInformation() { Clear(); }
			inline void Clear() {
				left_child_assignments = std::make_shared<AssignmentContainer>(false, 1);
				right_child_assignments = std::make_shared<AssignmentContainer>(false, 1);
			}
			std::shared_ptr<AssignmentContainer> left_child_assignments, right_child_assignments;
		};

		void InitialiseChildrenInfo();
		void InitializeBranches(const Branch& branch, const BinaryData& data);
		
		void UpdateBestLeftChild(int root_feature, const PartialSolution& solution, int upper_bound);
		void UpdateBestRightChild(int root_feature, const PartialSolution& solution, int upper_bound);
		void UpdateBestTwoNodeAssignment(const Branch& branch, int root_feature, int upper_bound);
		void UpdateBestThreeNodeAssignment(const Branch& branch, int root_feature, int upper_bound);
		void Merge(int feature, const Branch& branch, const Branch& left_branch, const Branch& right_branch, int& upper_bound,
			AssignmentContainer* left_solutions, AssignmentContainer* right_solutions, AssignmentContainer* final_solutions);

		TerminalResults results;
		std::vector<ChildrenInformation> best_children_info;
		std::vector<Branch> left_branches;
		std::vector<Branch> right_branches;
		FrequencyCounter frequency_counter;
		Solver* solver;
		int num_features;
	};

}