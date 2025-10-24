#include "model/partial_solution.h"

namespace DPF {

	double PartialSolution::GetBestDiscrimination(const Branch& branch, const DataSummary& data_summary) const {
		auto& budget = branch.GetDiscriminationBudget();
		const double a = GetInbalance(data_summary) + budget.min_balance;
		const double b = GetInbalance(data_summary) + budget.max_balance;
		if (a <= 0 && b >= 0) return 0;
		return std::min(std::abs(a), std::abs(b));
	}


	double PartialSolution::GetWorstDiscrimination(const Branch& branch, const DataSummary& data_summary) const {
		auto& budget = branch.GetDiscriminationBudget();
		const double a = GetInbalance(data_summary) + budget.min_balance;
		const double b = GetInbalance(data_summary) + budget.max_balance;
		return (a <= 0 && b <= 0 ? -a
			: (a >= 0 && b >= 0 ? b
				: (-a > b ? -a : b)));
	}

	void PartialSolution::UpdateBestAndWorstDiscrimination(const Branch& branch, const DataSummary& data_summary, double& worst, double& best, double& partial) const {
		auto& budget = branch.GetDiscriminationBudget();
		partial = GetInbalance(data_summary);
		const double a = partial + budget.min_balance;
		const double b = partial + budget.max_balance;
		if (a <= 0 && b >= 0) {
			best = 0;
		} else {
			best = a < 0 ? -b : a;
		}
		worst = (a <= 0 && b <= 0 ? -a
					: (a >= 0 && b >= 0 ? b
						: (-a > b ? -a : b)));
	}

	PartialSolution PartialSolution::Merge(const PartialSolution& sol1, const PartialSolution& sol2) {
		return PartialSolution(sol1.misclassifications + sol2.misclassifications,
			sol1.group0_positive + sol2.group0_positive, sol1.group1_positive + sol2.group1_positive);
	}
}