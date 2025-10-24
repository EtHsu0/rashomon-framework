/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "model/partial_solution.h"

namespace DPF {

	struct InternalNodeDescription;

	struct NodeCompare {
		NodeCompare() : misclassifications(INT32_MAX), worst_discrimination(0), best_discrimination(0), partial_discrimination(0) {}
		NodeCompare(int misclassifications, double worst_discrimination, double best_discrimination, double partial_discrimination) :
			misclassifications(misclassifications), worst_discrimination(worst_discrimination), 
			best_discrimination(best_discrimination), partial_discrimination(partial_discrimination) {
		}
		int misclassifications;
		double worst_discrimination;
		double best_discrimination;
		double partial_discrimination;

		inline bool operator==(const NodeCompare& b) const {
			return misclassifications == b.misclassifications &&
				std::abs(worst_discrimination - b.worst_discrimination) <= DISC_EPS &&
				std::abs(best_discrimination - b.best_discrimination) <= DISC_EPS &&
				std::abs(partial_discrimination - b.partial_discrimination) <= DISC_EPS;
		}
	};

	struct InternalNodeDescription {
		int feature;
		int label;
		PartialSolution solution;
		int num_nodes_left;
		int num_nodes_right;
		NodeCompare node_compare;

		InternalNodeDescription() : feature(INT32_MAX), label(INT32_MAX), num_nodes_left(INT32_MAX), num_nodes_right(INT32_MAX) {}
		InternalNodeDescription(int feature, int label, const PartialSolution& solution, int num_nodes_left, int num_nodes_right, const Branch& branch, const DataSummary& data_summary) :
			feature(feature), label(label), solution(solution), num_nodes_left(num_nodes_left), num_nodes_right(num_nodes_right) {
			node_compare.misclassifications = solution.GetMisclassifications();
			solution.UpdateBestAndWorstDiscrimination(branch, data_summary, node_compare.worst_discrimination, node_compare.best_discrimination, node_compare.partial_discrimination);
		}
		InternalNodeDescription(const InternalNodeDescription& node) = default;

		inline const PartialSolution& GetPartialSolution() const { return solution; }
		inline int NumNodes() const { return feature == INT32_MAX ? 0 : num_nodes_left + num_nodes_right + 1; }
		inline bool IsInfeasible() const { return !IsFeasible(); }
		inline bool IsFeasible() const { return solution.IsFeasible(); }
		inline bool NodeCompareInitialized() const { return node_compare.misclassifications != INT32_MAX; }
	
		inline int GetMisclassifications() const { return node_compare.misclassifications; }
		inline int GetObjectiveScore(const int sparsity_coefficient) const { return node_compare.misclassifications + NumNodes() * sparsity_coefficient; }
		inline double GetWorstDiscrimination() const { return node_compare.worst_discrimination; }
		inline double GetBestDiscrimination() const { return node_compare.best_discrimination; }
		
		void PrintSolution() const {
			if (feature == INT32_MAX)
				std::cout << "Leaf node (label=" << label << ")";
			else
				std::cout << "Branch node f" << feature;
			if (solution.IsFeasible())
				std::cout << ": " << solution;
			std::cout << std::endl;
		}
	};

	struct InternalTreeNode {
		InternalNodeDescription parent, left_child, right_child;
	};

}

namespace std {
	template <>
	struct hash<DPF::NodeCompare> {

		//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		size_t operator()(const DPF::NodeCompare& k) const {
			using std::size_t;
			using std::hash;

			size_t seed = k.misclassifications;
			seed ^= hash<int>()(int(std::floor(k.best_discrimination / DISC_EPS))) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hash<int>()(int(std::floor(k.worst_discrimination / DISC_EPS))) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}

	};
}