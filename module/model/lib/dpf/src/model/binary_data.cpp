/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "model/binary_data.h"


namespace DPF {

	OriginalData::OriginalData(const OriginalData& data) : labels(data.labels), groups(data.groups), num_features_(data.num_features_) {
		for (auto& fv : data.data) {
			this->data.push_back(new FeatureVectorBinary(*fv));
		}
	}

	/* FeatureVectors */
	void OriginalData::PrintStats() const {
		std::cout << "Instances: " << Size() << "\n";
		std::cout << "Num features: " << NumFeatures() << "\n";
	}

	double OriginalData::ComputeSparsity() const {
		if (Size() == 0) { return 0.0; }
		double sum_sparsity = 0.0;
		for (const FeatureVectorBinary* fv : data) {
			sum_sparsity += fv->Sparsity();
		}
		return sum_sparsity / (Size());
	}

	BinaryData::BinaryData(const BinaryData& other) : 
		hash_value_(other.hash_value_), is_closure_set_(other.is_closure_set_), 
		closure_(other.closure_), data(other.data)
	{
		for (int y = 0; y < 2; y++) {
			label_view[y] = other.label_view[y];
		}
		for (int a = 0; a < 2; a++) {
			group_view[a] = other.group_view[a];
		}
	}

	/* BinaryData */
	BinaryData::BinaryData(const OriginalData* data, const std::vector<int>(&label_indices)[2], const std::vector<int>(&group_indices)[2]) :
		data(data),
		hash_value_(-1),
		is_closure_set_(false) {
		
		for (int y = 0; y < 2; y++) {
			label_view[y] = DataView(data, label_indices[y]);
		}
		for (int a = 0; a < 2; a++) {
			group_view[a] = DataView(data, group_indices[a]);
		}
	}

	void BinaryData::SetClosure(const Branch& closure) {
		closure_ = closure;
		is_closure_set_ = true;
	}

	void BinaryData::SetHash(long long new_hash) {
		runtime_assert(new_hash != -1);
		hash_value_ = new_hash;
	}

	void BinaryData::SplitData(int feature, BinaryData& left, BinaryData& right) const {
		left.data = data;
		right.data = data;
		std::vector<int> label_indices[2][2]; // second index: 0=left, 1=right
		std::vector<int> group_indices[2][2]; // second index: 0=left, 1=right
		for (int i = 0; i < Size(); i++) {
			int ix = GetIndex(i);
			int label = data->GetLabel(ix);
			int group = data->GetGroup(ix);
			int direction = int(data->at(ix)->IsFeaturePresent(feature));
			label_indices[label][direction].push_back(ix);
			group_indices[group][direction].push_back(ix);
		}
		for (int j = 0; j < 2; j++) {
			left.label_view[j] = DataView(data, label_indices[j][0]);
			left.group_view[j] = DataView(data, group_indices[j][0]);
			right.label_view[j] = DataView(data, label_indices[j][1]);
			right.group_view[j] = DataView(data, group_indices[j][1]);
		}
	}

	void BinaryData::TrainTestSplitData(double test_percentage, BinaryData& train, BinaryData& test) const {
		runtime_assert(test_percentage >= 0.0 && test_percentage <= 1.0);
		train.data = data;
		test.data = data;
		std::vector<int> train_label_indices[2];
		std::vector<int> test_label_indices[2];
		std::vector<int> train_group_indices[2];
		std::vector<int> test_group_indices[2];
		std::vector<int> train_label_group_indices[2][2];
		std::vector<int> test_label_group_indices[2][2];

		std::vector<int> test_indices, train_indices;
		static auto rng = std::default_random_engine{};
		rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
		std::vector<int> all_label_group_indices[2][2];
		for (int i = 0; i < Size(); i++) {
			int label = data->GetLabel(i);
			int group = data->GetGroup(i);
			all_label_group_indices[label][group].push_back(i);
		}
		
		// Apply Stratification
		for (int y = 0; y < 2; y++) {
			for (int a = 0; a < 2; a++) {
				std::vector<int> indices;
				auto& label_group_indices = all_label_group_indices[y][a];
				indices.assign(label_group_indices.begin(), label_group_indices.end());
				std::shuffle(indices.begin(), indices.end(), rng);
				const int test_split_index = int(std::round(test_percentage * indices.size()));

				const auto begin = indices.begin();
				const auto split = indices.begin() + test_split_index;
				const auto end = indices.end();
				test_label_indices[y].insert(test_label_indices[y].end(), begin, split);
				test_group_indices[a].insert(test_group_indices[a].end(), begin, split);
				test_label_group_indices[y][a].insert(test_label_group_indices[y][a].end(), begin, split);
				train_label_indices[y].insert(train_label_indices[y].end(), split, end);
				train_group_indices[a].insert(train_group_indices[a].end(), split, end);
				train_label_group_indices[y][a].insert(train_label_group_indices[y][a].end(), split, end);
			}
		}
		
		for (int j = 0; j < 2; j++) {
			std::sort(train_label_indices[j].begin(), train_label_indices[j].end());
			std::sort(train_group_indices[j].begin(), train_group_indices[j].end());
			std::sort(test_label_indices[j].begin(), test_label_indices[j].end());
			std::sort(test_group_indices[j].begin(), test_group_indices[j].end());
			train.label_view[j] = DataView(data, train_label_indices[j]);
			train.group_view[j] = DataView(data, train_group_indices[j]);
			test.label_view[j] = DataView(data, test_label_indices[j]);
			test.group_view[j] = DataView(data, test_group_indices[j]);
		}
	}

	void BinaryData::AddInstance(int id) {
		int group = data->GetGroup(id);
		int label = data->GetLabel(id);
		label_view[label].Add(id);
		group_view[group].Add(id);
	}

	int BinaryData::GetLabelGroupCount(int label, int group) const {
		int count = 0;
		auto& instances = GetInstancesForLabel(label).GetIndices();
		for (auto ix : instances) {
			if (GetGroup(ix) == group) count++;
		}
		return count;
	}

	BinaryData* BinaryData::GetDeepCopy() const {
		OriginalData* o_data = new OriginalData(*(this->GetOriginalData()));
		std::vector<int> label_indices[2];
		std::vector<int> group_indices[2];
		for (int i = 0; i < 2; i++) {
			label_indices[i] = this->label_view[i].GetIndices();
			group_indices[i] = this->group_view[i].GetIndices();
		}
		return new BinaryData(o_data, label_indices, group_indices);
	}

	void BinaryData::PrintStats() const {
		data->PrintStats();
		//Print a contingency table
		std::cout << "        |";
		for (int label = 0; label < NumLabels(); label++) {
			std::cout << " label=" << label << " |";
		}
		std::cout << " Total   |" << std::endl;
		std::cout << "--------+---------+---------+---------+" << std::endl;
		for (int group = 0; group < NumGroups(); group++) {
			std::cout << "group=" << group << " |";
			for (int label = 0; label < NumLabels(); label++) {

				std::cout << std::setw(8) << GetLabelGroupCount(label, group) << " |";
			}
			std::cout <<std::setw(8) << group_view[group].Size() << " |" << std::endl;
		}
		std::cout << "--------+---------+---------+---------+" << std::endl;
		std::cout << "Total   |";
		for (int label = 0; label < NumLabels(); label++) {
			std::cout << std::setw(8) << label_view[label].Size() << " |";
		}
		std::cout << std::setw(8) << Size() << " |" << std::endl;
		std::cout << "Sparsity: " << ComputeSparsity() << std::endl;
	}

	std::ostream& operator<<(std::ostream& os, const BinaryData& data) {
		os << "Number of instances: " << data.Size();

		for (int label = 0; label < data.NumLabels(); label++) {
			int counter_instances = 0;
			os << std::endl << "Number of instances with label " << label << ": " << data.NumInstancesForLabel(label);
			for (auto fv : data.GetInstancesForLabel(label)) {
				os << std::endl << "\t" << counter_instances++ << ":\t" << fv;
			}
		}
		return os;
	}

}