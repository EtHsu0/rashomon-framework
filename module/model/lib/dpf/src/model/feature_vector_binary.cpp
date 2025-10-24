/**
From Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "model/feature_vector_binary.h"

namespace DPF {

	FeatureVectorBinary::FeatureVectorBinary(const std::vector<bool>& feature_values, int id) :
		id_(id),
		is_feature_present_(feature_values.size()) {
		for (size_t feature_index = 0; feature_index < feature_values.size(); feature_index++) {
			if (feature_values[feature_index] == true) { present_features_.push_back(feature_index); }
			is_feature_present_[feature_index] = feature_values[feature_index];
		}
	}

	double FeatureVectorBinary::Sparsity() const {
		return double(NumPresentFeatures()) / is_feature_present_.size();
	}

	std::ostream& operator<<(std::ostream& os, const FeatureVectorBinary& fv) {
		if (fv.NumPresentFeatures() == 0) { std::cout << "[empty]"; } else {
			auto iter = fv.begin();
			os << *iter;
			++iter;
			while (iter != fv.end()) {
				os << " " << *iter;
				++iter;
			}
		}
		return os;
	}

}