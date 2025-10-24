/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"

namespace DPF {

	struct DiscriminationBudget {
		double min_balance, max_balance;
		
		DiscriminationBudget() : min_balance(0.0), max_balance(0.0) {}
		DiscriminationBudget(double min_balance, double max_balance) :
			min_balance(min_balance), max_balance(max_balance) { }

		void Tighten(const DiscriminationBudget& parent, const DiscriminationBudget& other) {
			min_balance = std::max(min_balance, parent.min_balance + other.min_balance);
			max_balance = std::min(max_balance, parent.max_balance + other.max_balance);
		}

		bool IsRestricted() const {
			return std::abs(-1.0 - min_balance) > DBL_EPSILON
				|| std::abs(1.0 - max_balance) > DBL_EPSILON;
		}

		inline bool operator==(const DiscriminationBudget& rhs) const {
			return std::abs(min_balance - rhs.min_balance) < DISC_EPS
				&& std::abs(max_balance - rhs.max_balance) < DISC_EPS;
		}

		inline bool operator>=(const DiscriminationBudget& rhs) const {
			return min_balance <= rhs.min_balance + DISC_EPS
				&& max_balance >= rhs.max_balance - DISC_EPS;
		}

		inline bool IsFeasible() const {
			return min_balance <= max_balance + DISC_EPS;
		}

		static DiscriminationBudget nonRestrictedBudget;
	};

	class Branch {
	public:
		Branch() {}

		inline int Depth() const { return int(branch_codes_.size()); }
		inline int operator[](int i) const { return branch_codes_[i]; }
		inline const DiscriminationBudget& GetDiscriminationBudget() const { return budget; }
		inline void SetDiscriminationBudget(const DiscriminationBudget& budget) { this->budget = budget; }

		static Branch LeftChildBranch(const Branch& branch, int feature, const DiscriminationBudget& budget);
		static Branch RightChildBranch(const Branch& branch, int feature, const DiscriminationBudget& budget);

		bool HasBranchedOnFeature(int feature) const;

		bool operator==(const Branch& right_hand_side) const;
		friend std::ostream& operator<<(std::ostream& out, const Branch& branch) {
			for (int code : branch.branch_codes_) {
				out << code << " ";
			}
			return out;
		}

	private:
		inline int GetCode(int feature, bool present) const { return 2 * feature + present; }
		void AddFeatureBranch(int feature, bool present);
		void ConvertIntoCanonicalRepresentation();

		std::vector<int> branch_codes_;
		DiscriminationBudget budget;
	};
}