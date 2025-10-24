/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/feature_selector.h"

namespace DPF {

	void FeatureSelectorGini::InitializeInternal(const BinaryData& data) {

		uint32_t data_size = data.Size();

		//clear helper data structures
		double max_gini_value = -1.0;
		const int num_labels = data.NumLabels();

		std::vector<std::vector<int> > num_label_with_feature(num_labels, std::vector<int>(num_features));
		std::vector<std::vector<int> > num_label_without_this_feature(num_labels, std::vector<int>(num_features));
		std::vector<int> num_with_feature(num_features);
		std::vector<int> num_without_feature(num_features);
		std::vector<double> gini_values(num_features);

		for (int label = 0; label < num_labels; label++) {
			for (int feature = 0; feature < num_features; feature++) {
				num_label_with_feature[label][feature] = 0;
				num_label_without_this_feature[label][feature] = 0;
			}
		}

		for (int feature = 0; feature < num_features; feature++) { 
			gini_values[feature] = 0.0;
			num_with_feature[feature] = 0;
			num_without_feature[feature] = 0;
		}

		for (int i = 0; i < data.Size(); i++) {
			int label = data.GetLabel(i);
			for (int feature = 0; feature < num_features; feature++) {
				if (data.GetInstance(i)->IsFeaturePresent(feature)) {
					num_label_with_feature[label][feature]++;
					num_with_feature[feature]++;
				} else {
					num_label_without_this_feature[label][feature]++;
					num_without_feature[feature]++;
				}
			}

		}

		//compute the gini values for each feature 
		double I_D = 1.0;
		for (int label = 0; label < num_labels; label++) {
			I_D -= pow(double(data.NumInstancesForLabel(label)) / data.Size(), 2);
		}
		
		for (int feature = 0; feature < num_features; feature++) {

			double I_D_without_feature = 1.0;
			if (num_without_feature[feature] > 0) {
				for (int label = 0; label < num_labels; label++) {
					I_D_without_feature -= pow(double(num_label_without_this_feature[label][feature]) / num_without_feature[feature], 2);
				}
			}

			double I_D_with_feature = 1.0;
			if (num_with_feature[feature] > 0) {
				for (int label = 0; label < num_labels; label++) {
					I_D_with_feature -= pow(double(num_label_with_feature[label][feature]) / num_with_feature[feature], 2);
				}
			}

			gini_values[feature] = I_D - (double(num_without_feature[feature]) / data_size) * I_D_without_feature
				- (double(num_with_feature[feature]) / data_size) * I_D_with_feature;
			
			max_gini_value = std::max(gini_values[feature], max_gini_value);

		}

		for (int feature = 0; feature < num_features; feature++) {
			feature_order[feature] = gini_values[feature];
		}

	}
}