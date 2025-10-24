/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/

#include "solver/difference_computer.h"

namespace DPF {
	DifferenceMetrics BinaryDataDifferenceComputer::ComputeDifferenceMetrics(const BinaryData& data_old, const BinaryData& data_new) {
		DifferenceMetrics metrics;
		
		for (int label = 0; label < data_new.NumLabels(); label++) {
			auto& new_instances = data_new.GetInstancesForLabel(label);
			auto& old_instances = data_old.GetInstancesForLabel(label);
			int size_new = new_instances.Size();
			int size_old = old_instances.Size();
			int index_new = 0, index_old = 0;
			while (index_new < size_new && index_old < size_old) {
				int id_new = new_instances.GetIndex(index_new);
				int id_old = old_instances.GetIndex(index_old);

				//the new data has something the old one does not
				if (id_new < id_old) {
					metrics.total_difference++;
					int group = data_new.GetOriginalData()->GetGroup(id_new);
					if (group == 1)
						metrics.num_additions_group1++;
					else metrics.num_additions_group0++;
					index_new++;
				}
				//the old data has something the new one does not
				else if (id_new > id_old) {
					metrics.total_difference++;
					metrics.num_removals++;
					int group = data_old.GetOriginalData()->GetGroup(id_old);
					if (group == 1)
						metrics.num_removals_group1++;
					else metrics.num_removals_group0++;
					index_old++;
				} else {//no difference
					index_new++;
					index_old++;
				}
			}
			metrics.total_difference += (size_new - index_new);
			metrics.total_difference += (size_old - index_old);
			metrics.num_removals += (size_old - index_old);
			for (; index_new < size_new; index_new++) {
				int id_new = new_instances.GetIndex(index_new);
				int group = data_new.GetOriginalData()->GetGroup(id_new);
				if (group == 1)
					metrics.num_additions_group1++;
				else metrics.num_additions_group0++;
			}
			for (; index_old < size_old; index_old++) {
				int id_old = old_instances.GetIndex(index_old);
				int group = data_old.GetOriginalData()->GetGroup(id_old);
				if (group == 1)
					metrics.num_removals_group1++;
				else metrics.num_removals_group0++;
			}
		}
		return metrics;
	}

	DifferenceMetrics BinaryDataDifferenceComputer::ComputeDifferenceMetricsWithoutGroups(const BinaryData& data_old, const BinaryData& data_new) {
		DifferenceMetrics metrics;

		for (int label = 0; label < data_new.NumLabels(); label++) {
			auto& new_instances = data_new.GetInstancesForLabel(label);
			auto& old_instances = data_old.GetInstancesForLabel(label);
			int size_new = new_instances.Size();
			int size_old = old_instances.Size();
			int index_new = 0, index_old = 0;
			while (index_new < size_new && index_old < size_old) {
				int id_new = new_instances.GetIndex(index_new);
				int id_old = old_instances.GetIndex(index_old);

				//the new data has something the old one does not
				if (id_new < id_old) {
					metrics.total_difference++;
					index_new++;
				}
				//the old data has something the new one does not
				else if (id_new > id_old) {
					metrics.total_difference++;
					metrics.num_removals++;
					index_old++;
				} else {//no difference
					index_new++;
					index_old++;
				}
			}
			metrics.total_difference += (size_new - index_new);
			metrics.total_difference += (size_old - index_old);
			metrics.num_removals += (size_old - index_old);
		}
		return metrics;
	}

	void BinaryDataDifferenceComputer::ComputeDifference(const BinaryData& data_old, const BinaryData& data_new, BinaryData& data_to_add, BinaryData& data_to_remove) {

		for (int label = 0; label < data_new.NumLabels(); label++) {
			auto& new_instances = data_new.GetInstancesForLabel(label);
			auto& old_instances = data_old.GetInstancesForLabel(label);
			int size_new = new_instances.Size();
			int size_old = old_instances.Size();
			int index_new = 0, index_old = 0;
			int id_new_prev = -1;
			int id_old_prev = -1;
			while (index_new < size_new && index_old < size_old) {
				int id_new = new_instances.GetIndex(index_new);
				int id_old = old_instances.GetIndex(index_old);
				runtime_assert(id_new_prev <= id_new && id_old_prev <= id_old);
				id_new_prev = id_new;
				id_old_prev = id_old;
				//the new data has something the old one does not
				if (id_new < id_old) {
					data_to_add.AddInstance(id_new);
					index_new++;
				}
				//the old data has something the new one does not
				else if (id_new > id_old) {
					data_to_remove.AddInstance(id_old);
					index_old++;
				} else {//no difference
					index_new++;
					index_old++;
				}
			}//end while

			for (; index_new < size_new; index_new++) {
				int id_new = new_instances.GetIndex(index_new);
				data_to_add.AddInstance(id_new);
			}

			for (; index_old < size_old; index_old++) {
				int id_old = old_instances.GetIndex(index_old);
				data_to_remove.AddInstance(id_old);
			}
		}
	}

}
