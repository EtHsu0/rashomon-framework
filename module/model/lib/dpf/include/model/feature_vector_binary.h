/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"

namespace DPF {
	class FeatureVectorBinary {
	public:
		FeatureVectorBinary(const std::vector<bool>& feature_values, int id);

		inline bool IsFeaturePresent(const int feature) const { return is_feature_present_[feature]; }
		inline int GetJthPresentFeature(const int j) const { return present_features_[j]; }
		inline int NumPresentFeatures() const { return int(present_features_.size()); }
		inline int NumTotalFeatures() const { return int(is_feature_present_.size()); }
		inline int GetID() const { return id_; }
		double Sparsity() const;

		inline std::vector<int>::const_iterator begin() const { return present_features_.begin(); }
		inline std::vector<int>::const_iterator end() const { return present_features_.end(); }

		friend std::ostream& operator<<(std::ostream& os, const FeatureVectorBinary& fv);

	private:
		int id_;
		std::vector<char> is_feature_present_; //[i] indicates if the feature is true or false, i.e., if it is present in present_Features.
		std::vector<int> present_features_;

	};
}