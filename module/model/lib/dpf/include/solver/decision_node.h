/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"

namespace DPF {

	struct DecisionNode {
		
		static std::shared_ptr<DecisionNode> CreateLabelNode(int label) {
			runtime_assert(label != INT32_MAX);
			auto node = std::make_shared<DecisionNode>();
			node->feature = INT32_MAX;
			node->label = label;
			node->left_child = nullptr;
			node->right_child = nullptr;
			return node;
		}

		static std::shared_ptr<DecisionNode> CreateFeatureNodeWithNullChildren(int feature) {
			runtime_assert(feature != INT32_MAX);
			auto node = std::make_shared<DecisionNode>();
			node->feature = feature;
			node->label = INT32_MAX;
			node->left_child = NULL;
			node->right_child = NULL;
			return node;
		}

		int Depth() const {
			if (IsLabelNode()) { return 0; }
			return 1 + std::max(left_child->Depth(), right_child->Depth());
		}

		int NumNodes() const {
			if (IsLabelNode()) { return 0; }
			return 1 + left_child->NumNodes() + right_child->NumNodes();
		}

		int PrintTree(std::ofstream &myfile) const {
			if (IsLabelNode()) {
				myfile << "{ \"prediction\": " << label << "} ";
				return 1;
			}
			myfile << "{ " << "\"feature\": " << feature << ", ";
			myfile << " \"false\": ";
			left_child->PrintTree(myfile);
			myfile << ", ";
			myfile << " \"true\": ";
			right_child->PrintTree(myfile);
			myfile << " }";
			return 1;
		}

		bool IsLabelNode() const { return label != INT32_MAX; }
		bool IsFeatureNode() const { return feature != INT32_MAX; }

		int ComputeMisclassificationScore(const BinaryData& data) const {
			int misclassifications = 0;
			for (int label = 0; label < data.NumLabels(); label++) {
				for (const auto& fv : data.GetInstancesForLabel(label)) {
					misclassifications += (Classify(fv) != label);
				}
			}
			return misclassifications;
		}

		double ComputeDiscriminationScore(const BinaryData& data) const {
			int pos_group0 = 0, pos_group1 = 0;
			for (auto& fv : data.GetInstancesForGroup(0))
				pos_group0 += CountPositives(fv);
			for (auto& fv : data.GetInstancesForGroup(1))
				pos_group1 += CountPositives(fv);
			int pos_total = pos_group0 + pos_group1;
			double p = double(pos_total) / double(data.Size());
			if (p <= DBL_EPSILON) return 0;
			return (double(pos_group0) / double(data.NumInstancesForGroup(0)) -
				double(pos_group1) / double(data.NumInstancesForGroup(1)));
		}

		double ComputeGroupPerformance(const BinaryData& data, int group) const {
			int pos_group = 0;
			for (auto& fv : data.GetInstancesForGroup(group))
				pos_group += CountPositives(fv);
			return double(pos_group) / double(data.NumInstancesForGroup(group));
		}

		int Classify(const FeatureVectorBinary& feature_vector) const {
			if (IsLabelNode()) {
				return label;
			} else if (feature_vector.IsFeaturePresent(feature)) {
				return right_child->Classify(feature_vector);
			} else {
				return left_child->Classify(feature_vector);
			}
		}

		int CountPositives(const FeatureVectorBinary& feature_vector) const {
			if (IsLabelNode()) {
				return (label == 1) ? 1 : 0;
			} else if (feature_vector.IsFeaturePresent(feature)) {
				return right_child->CountPositives(feature_vector);
			} else {
				return left_child->CountPositives(feature_vector);
			}
		}

		int feature, label;
		std::shared_ptr<DecisionNode> left_child, right_child;
	};
}