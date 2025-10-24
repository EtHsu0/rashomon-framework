/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "model/binary_data.h"

namespace DPF {

	class FeatureSelectorAbstract {
	public:
		FeatureSelectorAbstract() = delete;
		FeatureSelectorAbstract(int num_features) : num_features(num_features), num_features_popped(0) {}
		virtual ~FeatureSelectorAbstract() {}
		
		void Initialize(const BinaryData& data) {
			InitializeInternal(data);
		}

		int PopNextFeature() {
			runtime_assert(AreThereAnyFeaturesLeft());
			int next_feature = PopNextFeatureInternal();
			num_features_popped++;
			return next_feature;
		}
		inline bool AreThereAnyFeaturesLeft() const { return num_features_popped != num_features;  }

	protected:
		virtual int PopNextFeatureInternal() = 0;
		virtual void InitializeInternal(const BinaryData& data) = 0;

		int num_features;
		int num_features_popped;
	};

	class FeatureSelectorInOrder : public FeatureSelectorAbstract {
	public:
		FeatureSelectorInOrder() = delete;
		FeatureSelectorInOrder(int num_features) : FeatureSelectorAbstract(num_features), next(0) {}
		~FeatureSelectorInOrder() = default;
	protected:
		int PopNextFeatureInternal() { return next++;  }
		void InitializeInternal(const BinaryData& data) {}
	private:
		int next;
	};

	class FeatureSelectorGini : public FeatureSelectorAbstract {
	public:
		FeatureSelectorGini() = delete;
		FeatureSelectorGini(int num_features) : FeatureSelectorAbstract(num_features) {}
		~FeatureSelectorGini() = default;
	protected:
		int PopNextFeatureInternal() {
			auto it = feature_order.begin();
			int feature = it->first;
			feature_order.erase(it);
			return feature;
		}
		void InitializeInternal(const BinaryData& data);

	private:
		std::map<int, double> feature_order;
	};
}