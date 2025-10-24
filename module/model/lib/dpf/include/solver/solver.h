/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "utils/file_reader.h"
#include "utils/parameter_handler.h"
#include "utils/stopwatch.h"
#include "model/internal_node_description.h"
#include "model/container.h"
#include "solver/solver_result.h"
#include "solver/feature_selector.h"
#include "solver/abstract_cache.h"
#include "solver/dataset_cache.h"
#include "solver/branch_cache.h"
#include "solver/decision_node.h"
#include "solver/terminal_solver.h"
#include "solver/statistics.h"
#include "solver/similarity_lowerbound.h"

namespace DPF {
	class Solver {
	public:
		Solver(ParameterHandler& solver_parameters);
		~Solver();

		const SolverResult HyperSolve();
		const SolverResult Solve();

		inline void SetVerbosity(bool verbose) { verbose_ = verbose; }
		void SetData(const BinaryData* data);

		inline const DataSummary& GetDataSummary() const { return data_summary; }
		void SplitTrainTestData();

		std::shared_ptr<DecisionNode> ConstructOptimalTree(const InternalNodeDescription& node, BinaryData& data, Branch& branch, int max_depth, int num_nodes); //reconstructs the optimal tree after the algorithm populated the cache

		std::shared_ptr<AssignmentContainer> SolveSubtree(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound);
		std::shared_ptr<AssignmentContainer> SolveSubtreeGeneralCase(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound);
		std::shared_ptr<AssignmentContainer> SolveTerminalNode(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound);
		bool UpdateCacheUsingSimilarity(BinaryData& data, Branch& branch, int max_depth, int num_nodes, int upper_bound);//returns true if the method updated the optimal solution of the branch; otherwise it only (tries) to increase the lower bound

		std::shared_ptr<AssignmentContainer> CreateLeafNodeDescriptions(const BinaryData& data, const Branch& branch, int accuracy_upper_bound) const;
		PartialSolution CreatePartialSolution(const BinaryData& data, int label) const;
				DiscriminationBudget GetDiscriminationBudget(const BinaryData& data, const Branch& branch) const;
		DiscriminationBudget GetDiscriminationBudget(int group0, int group1, const Branch& branch) const;
		void UpdateDiscriminationBudget(const Branch& org_branch, Branch& sub_branch, AssignmentContainer* partial_solutions) const;
		
		inline bool SatisfiesConstraint(const InternalNodeDescription& node, const Branch& branch) const { return SatisfiesConstraint(node.GetPartialSolution(), branch); }
		inline bool SatisfiesConstraint(const InternalNodeDescription& node) const { 
			runtime_assert(node.NodeCompareInitialized());
			return node.GetBestDiscrimination() <= discrimination_cutoff_value;
		}
		bool SatisfiesConstraint(const PartialSolution& solution, const Branch& branch) const;
		
		template<bool reconstruct>
		void Merge(int feature, const Branch& branch, const Branch& left_branch, const Branch& right_branch, int& upper_bound,
			AssignmentContainer* left_solutions, AssignmentContainer* right_solutions, AssignmentContainer* final_solutions, InternalTreeNode* tree_node=nullptr);
		void MergeAdd(int feature, const Branch& branch, int& upper_bound, const InternalNodeDescription& n1, const InternalNodeDescription& n2, AssignmentContainer* final_solutions);
		bool CheckSolution(const InternalNodeDescription& n1, const InternalNodeDescription& n2, InternalTreeNode* tree_node);

		inline bool IsTerminalNode(int depth, int num_nodes) { return depth <= 2; }
		inline int GetMinimumLeafNodeSize() const { return minimum_leaf_node_size; }
		inline int GetSparsityCoefficient() const { return sparsity_coefficient; }
		inline Statistics& GetStatistics() { return stats; }	

		void Reset();
	private:
		ParameterHandler parameters;
		int num_labels_, num_features; //for convenience, these are set upon instance creation and not changed
		int minimum_leaf_node_size;
		double discrimination_cutoff_value;
		int sparsity_coefficient;
		DataSummary data_summary;
		bool return_pareto_front;
		bool verbose_;
		AbstractCache* cache;
		BinaryData* binary_data;
		BinaryData train_data, test_data;
		Statistics stats;
		Stopwatch stopwatch_;
		SimilarityLowerBoundComputer* similarity_lower_bound_computer;
		TerminalSolver* terminal_solver1, *terminal_solver2;
	};
}