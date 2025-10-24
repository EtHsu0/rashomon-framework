/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "solver/terminal_solver.h"

namespace DPF {

	FrequencyCounter::FrequencyCounter(int num_features) :
		counts_group0(num_features),
		counts_group1(num_features)
	{
		
	}

	void FrequencyCounter::Initialize(const BinaryData& data) {
		const bool using_incremental_updates = this->data.IsInitialized() && true;
		BinaryData data_add(data.GetOriginalData()), data_remove(data.GetOriginalData());
		if (using_incremental_updates) {
			BinaryDataDifferenceComputer::ComputeDifference(this->data, data, data_add, data_remove);
		}
		if(using_incremental_updates && data_add.Size() + data_remove.Size() < data.Size()) {
			UpdateCounts(data_add, +1);
			UpdateCounts(data_remove, -1);
		} else {
			counts_group0.ResetToZeros();
			counts_group1.ResetToZeros();
			UpdateCounts(data, +1);
		}
		this->data = data;
	}

	int FrequencyCounter::ProbeDifference(const BinaryData& data) const {
		return BinaryDataDifferenceComputer::ComputeDifferenceMetricsWithoutGroups(this->data, data).total_difference;
	}

	void FrequencyCounter::UpdateCounts(const BinaryData& data, int value) {
		Counter* counters[] = { &counts_group0, &counts_group1 };
		for(int d_ix = 0; d_ix < data.Size(); d_ix++) {
			const auto data_point = data.GetInstance(d_ix);
			const int num_present_features = data_point->NumPresentFeatures();
			const int group = data.GetGroup(d_ix);
			const int label = data.GetLabel(d_ix);
			Counter* counter = counters[group];
			for (int i = 0; i < num_present_features; i++) {
				const int feature1 = data_point->GetJthPresentFeature(i);
				for (int j = i; j < num_present_features; j++) {
					const int feature2 = data_point->GetJthPresentFeature(j);
					runtime_assert(feature1 <= feature2);
					counter->CountLabel(label, feature1, feature2) += value;
					runtime_assert(counter->CountLabel(label, feature1, feature2) >= 0);
				}
			}
		}
	}

	// Positives
	int FrequencyCounter::PositivesZeroZero(int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return data.NumInstancesForLabel(1) 
			-  (PositivesOneOne(f1, f1) 
				+ PositivesOneOne(f2, f2) 
				- PositivesOneOne(f1, f2));
	}

	int FrequencyCounter::PositivesZeroOne(int f1, int f2) const {
		if (f1 > f2) { return PositivesOneZero(f2, f1); }
		return (counts_group0.Positives(f2, f2) + counts_group1.Positives(f2, f2)) -
			(counts_group0.Positives(f1, f2) + counts_group1.Positives(f1, f2));
	}

	int FrequencyCounter::PositivesOneZero(int f1, int f2) const {
		if (f1 > f2) { return PositivesZeroOne(f2, f1); }
		return (counts_group0.Positives(f1, f1) + counts_group1.Positives(f1, f1)) - 
			(counts_group0.Positives(f1, f2) + counts_group1.Positives(f1, f2));
	}

	int FrequencyCounter::PositivesOneOne(int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return counts_group0.Positives(f1, f2) + counts_group1.Positives(f1, f2);
	}

	// Negatives
	int FrequencyCounter::NegativesZeroZero(int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return data.NumInstancesForLabel(0)
			- (NegativesOneOne(f1, f1)
				+ NegativesOneOne(f2, f2)
				- NegativesOneOne(f1, f2));
	}

	int FrequencyCounter::NegativesZeroOne(int f1, int f2) const {
		if (f1 > f2) { return NegativesOneZero(f2, f1); }
		return (counts_group0.Negatives(f2, f2) + counts_group1.Negatives(f2, f2)) -
			(counts_group0.Negatives(f1, f2) + counts_group1.Negatives(f1, f2));
	}

	int FrequencyCounter::NegativesOneZero(int f1, int f2) const {
		if (f1 > f2) { return NegativesZeroOne(f2, f1); }
		return (counts_group0.Negatives(f1, f1) + counts_group1.Negatives(f1, f1)) -
			(counts_group0.Negatives(f1, f2) + counts_group1.Negatives(f1, f2));
	}

	int FrequencyCounter::NegativesOneOne(int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return counts_group0.Negatives(f1, f2) + counts_group1.Negatives(f1, f2);
	}

	// Group counts
	int FrequencyCounter::GroupZeroZero(int a, int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return data.NumInstancesForGroup(a) -
			(GroupOneOne(a, f1, f1) +
				GroupOneOne(a, f2, f2) -
				GroupOneOne(a, f1, f2));
	}

	int FrequencyCounter::GroupZeroOne(int a, int f1, int f2) const {
		if (f1 > f2) { return GroupOneZero(a, f2, f1); }
		return a == 0 ?
			counts_group0.Positives(f2, f2) + counts_group0.Negatives(f2, f2) -
			(counts_group0.Positives(f1, f2) + counts_group0.Negatives(f1, f2)) :
			counts_group1.Positives(f2, f2) + counts_group1.Negatives(f2, f2) -
			(counts_group1.Positives(f1, f2) + counts_group1.Negatives(f1, f2));
	}

	int FrequencyCounter::GroupOneZero(int a, int f1, int f2) const {
		if (f1 > f2) { return GroupZeroOne(a, f2, f1); }
		return a == 0 ?
			counts_group0.Positives(f1, f1) + counts_group0.Negatives(f1, f1) -
			(counts_group0.Positives(f1, f2) + counts_group0.Negatives(f1, f2)) :
			counts_group1.Positives(f1, f1) + counts_group1.Negatives(f1, f1) -
			(counts_group1.Positives(f1, f2) + counts_group1.Negatives(f1, f2));
	}

	int FrequencyCounter::GroupOneOne(int a, int f1, int f2) const {
		if (f1 > f2) { std::swap(f1, f2); }
		return a == 0 ?
			counts_group0.Positives(f1, f2) + counts_group0.Negatives(f1, f2) :
			counts_group1.Positives(f1, f2) + counts_group1.Negatives(f1, f2);
	}
}