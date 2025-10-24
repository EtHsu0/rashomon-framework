#pragma once
#include "base.h"
#include "model/branch.h"
#include "model/binary_data.h"

namespace DPF {

	class PartialSolution {
	public:
		PartialSolution() : PartialSolution(INT32_MAX, 0, 0) {}
		PartialSolution(int misclassifications, int group0_positive, int group1_positive) : 
			misclassifications(misclassifications), group0_positive(group0_positive), group1_positive(group1_positive) { 
		}
		inline int GetMisclassifications() const { return misclassifications; }
		inline double GetInbalance(const DataSummary& data_summary) const { 
			return double(group0_positive) / data_summary.group0_size - double(group1_positive) / data_summary.group1_size;
		}
		inline bool IsFeasible() const { return misclassifications < INT32_MAX; }
		inline int GetGroupBalance() const { return group0_positive - group1_positive; }
		double GetWorstDiscrimination(const Branch& branch, const DataSummary& data_summary) const;
		double GetBestDiscrimination(const Branch& branch, const DataSummary& data_summary) const;
		void UpdateBestAndWorstDiscrimination(const Branch& branch, const DataSummary& data_summary, double& worst, double& best, double& partial) const;

		friend std::ostream& operator<<(std::ostream& os, const PartialSolution& solution) {
			os << "Misclassifications: " << std::setw(8) << solution.GetMisclassifications() 
				<< ", Group 0 Pos.: " << std::setw(8) << solution.group0_positive
				<< ", Group 1 Pos.: " << std::setw(8) << solution.group1_positive;
			return os;
		}
		inline bool operator ==(const PartialSolution& b) const {
			return misclassifications == b.misclassifications &&
				group0_positive == b.group0_positive &&
				group1_positive == b.group1_positive;
		}

		static PartialSolution Merge(const PartialSolution& sol1, const PartialSolution& sol2);
	//private:
		int misclassifications;
		int group0_positive;
		int group1_positive;
	};
}

namespace std {
	template <>
	struct hash<DPF::PartialSolution> {

		//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		size_t operator()(const DPF::PartialSolution& k) const {
			using std::size_t;
			using std::hash;

			size_t seed = k.misclassifications;
			seed ^= hash<int>()(k.group0_positive) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= hash<int>()(k.group1_positive) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}

	};
}