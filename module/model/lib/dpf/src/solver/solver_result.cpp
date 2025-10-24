#include "solver/solver_result.h"

namespace DPF {

	const std::vector<InternalNodeDescription> SolverResult::GetSolutionsInOrder() const {
		std::vector<InternalNodeDescription> solutions(this->solutions->solutions.begin(), this->solutions->solutions.end());
		std::sort(solutions.begin(), solutions.end(), [](const InternalNodeDescription& n1, const InternalNodeDescription& n2) {
			return n1.GetPartialSolution().GetMisclassifications() < n2.GetPartialSolution().GetMisclassifications();
		});
		return solutions;
	}

	const Performance& SolverResult::GetPerformanceByMisclassificationScore(int misclassifications) const {
		for (auto& p : performances) {
			if (p.train_misclassifications == misclassifications) return p;
		}
		runtime_assert(1 == 0);
	}

}