/**
From Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "model/branch.h"

namespace DPF {

	DiscriminationBudget DiscriminationBudget::nonRestrictedBudget = { -1.0, 1.0 };

	void Branch::AddFeatureBranch(int feature, bool present) {
		int code = GetCode(feature, present);
		branch_codes_.push_back(code);

		ConvertIntoCanonicalRepresentation();
	}

	Branch Branch::LeftChildBranch(const Branch& branch, int feature, const DiscriminationBudget& budget) {
		Branch left_child_branch(branch);
		left_child_branch.AddFeatureBranch(feature, false); //the convention is that the left branch does not have the feature
		left_child_branch.SetDiscriminationBudget(budget);
		return left_child_branch;
	}

	Branch Branch::RightChildBranch(const Branch& branch, int feature, const DiscriminationBudget& budget) {
		Branch right_child_branch(branch);
		right_child_branch.AddFeatureBranch(feature, true); //the convention is that the right branch has the feature
		right_child_branch.SetDiscriminationBudget(budget);
		return right_child_branch;
	}

	bool Branch::operator==(const Branch& right_hand_side) const {
		if (this->branch_codes_.size() != right_hand_side.branch_codes_.size()) { return false; }
		for (size_t i = 0; i < this->branch_codes_.size(); i++) {
			if (this->branch_codes_[i] != right_hand_side.branch_codes_[i]) { return false; }
		}
		return true;
	}

	void Branch::ConvertIntoCanonicalRepresentation() {
		std::sort(branch_codes_.begin(), branch_codes_.end());
	}

	bool Branch::HasBranchedOnFeature(int feature) const {
		int code0 = GetCode(feature, true);
		int code1 = GetCode(feature, false);
		for (int i = 0; i < branch_codes_.size(); i++) {
			if (branch_codes_[i] == code0 || branch_codes_[i] == code1) return true;
		}
		return false;
	}

}
