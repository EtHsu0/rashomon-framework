/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/

#pragma once

#include "model/binary_data.h"

namespace DPF {

	struct DifferenceMetrics {
		DifferenceMetrics() :num_removals(0), total_difference(0),
			num_removals_group0(0), num_removals_group1(0), num_additions_group0(0), num_additions_group1(0) {}
		int num_removals, total_difference,
			num_removals_group0, num_removals_group1,
			num_additions_group0, num_additions_group1;
	};

	class BinaryDataDifferenceComputer {
	public:
		static DifferenceMetrics ComputeDifferenceMetrics(const BinaryData& data_old, const BinaryData& data_new);
		static DifferenceMetrics ComputeDifferenceMetricsWithoutGroups(const BinaryData& data_old, const BinaryData& data_new);
		static void ComputeDifference(const BinaryData& data_old, const BinaryData& data_new, BinaryData& data_to_add, BinaryData& data_to_remove);
	};
}