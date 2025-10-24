#pragma once
#include "model/container.h"
#include "solver/decision_node.h"

namespace DPF {

	struct Performance {
		int train_misclassifications,
			test_misclassifications;
		double train_discrimination,
			test_discrimination,
			train_accuracy,
			test_accuracy,
			test_performance_group0,
			test_performance_group1;

		static Performance ComputePerformance(const DecisionNode* tree, const BinaryData& train_data, BinaryData& test_data) {
			Performance result;
			result.train_misclassifications = tree->ComputeMisclassificationScore(train_data);
			result.train_discrimination = tree->ComputeDiscriminationScore(train_data);
			result.train_accuracy = 1.0 - double(result.train_misclassifications) / train_data.Size();
			if (test_data.Size() > 0) {
				result.test_misclassifications = tree->ComputeMisclassificationScore(test_data);
				result.test_discrimination = tree->ComputeDiscriminationScore(test_data);
				result.test_performance_group0 = tree->ComputeGroupPerformance(test_data, 0);
				result.test_performance_group1 = tree->ComputeGroupPerformance(test_data, 1);
				result.test_accuracy = 1.0 - double(result.test_misclassifications) / test_data.Size();
			} else {
				result.test_misclassifications = -1;
				result.test_discrimination = 0;
				result.test_accuracy = 0;
				result.test_performance_group0 = 0;
				result.test_performance_group1 = 0;
			}
			return result;
		}

		static Performance GetAverage(const std::vector<Performance>& performances) {
			Performance result({ 0,0,0,0,0,0 });
			if (performances.size() == 0) return result;
			for (const auto& p : performances) {
				result.train_misclassifications += p.train_misclassifications;
				result.train_discrimination += p.train_discrimination;
				result.train_accuracy += p.train_accuracy;
				result.test_misclassifications += p.test_misclassifications;
				result.test_discrimination += p.test_discrimination;
				result.test_accuracy += p.test_accuracy;
				result.test_performance_group0 += p.test_performance_group0;
				result.test_performance_group1 += p.test_performance_group1;
			}
			result.train_misclassifications = double(result.train_misclassifications) / performances.size();
			result.train_discrimination = double(result.train_discrimination) / performances.size();
			result.train_accuracy = double(result.train_accuracy) / performances.size();
			result.test_misclassifications = double(result.test_misclassifications) / performances.size();
			result.test_discrimination = double(result.test_discrimination) / performances.size();
			result.test_accuracy = double(result.test_accuracy) / performances.size();
			result.test_performance_group0 = double(result.test_performance_group0) / performances.size();
			result.test_performance_group1 = double(result.test_performance_group1) / performances.size();
			return result;
		}
	};

	struct SolverResult {
	
		SolverResult() = default;
		SolverResult(std::shared_ptr<AssignmentContainer> solutions, bool is_proven_optimal)
			: solutions(solutions), is_proven_optimal(is_proven_optimal) {}

		inline bool IsFeasible() const { return solutions->Size() > 0; }
		inline bool IsProvenOptimal() const { return is_proven_optimal; }
		inline const std::shared_ptr<AssignmentContainer> GetSolutions() const { return solutions; }
		const std::vector<InternalNodeDescription> GetSolutionsInOrder() const;
		const Performance& GetPerformanceByMisclassificationScore(int misclassifications) const;
		int PrintAllTree(std::ofstream &myfile) const {
			myfile << "{ ";
			for(unsigned int i = 0; i < trees.size(); i++) {
				myfile << "\"" << i << "\": ";
				trees[i]->PrintTree(myfile);
				if ((i+1)!=trees.size()) myfile << ", ";
			}
			myfile << " }";
			return 1;
		}

		bool is_proven_optimal;
		std::shared_ptr<AssignmentContainer> solutions;
		std::vector<std::shared_ptr<DecisionNode>> trees;
		std::vector<Performance> performances;
	};

}