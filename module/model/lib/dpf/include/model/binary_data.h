/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "model/feature_vector_binary.h"
#include "model/branch.h"


namespace DPF {

	class OriginalData {
	public:
		OriginalData(const std::vector<const FeatureVectorBinary*>& data, const std::vector<int>& labels, const std::vector<int>& groups) :
			data(data), labels(labels), groups(groups),
			num_features_(data.size() > 0 ? data.at(0)->NumTotalFeatures() : 0) {}
		OriginalData(const OriginalData& data);
		~OriginalData() { Clear(); }
		inline void Clear() {
			for (auto fv : data) delete fv;
			data.clear();
		}
		inline const FeatureVectorBinary* at(int ix) const { return data[ix]; }
		inline int NumFeatures() const { return num_features_; }
		inline int Size() const { return int(data.size()); }
		inline int GetLabel(int ix) const { return labels.at(ix); }
		inline int GetGroup(int ix) const { return groups.at(ix); }
		double ComputeSparsity() const;
		void PrintStats() const;
	private:
		std::vector<const FeatureVectorBinary*> data;
		std::vector<int> labels;
		std::vector<int> groups;
		int num_features_;
	};

	class DataView {
	public:
		struct Iterator {
			using iterator_category = std::forward_iterator_tag;
			using value_type = const FeatureVectorBinary*;
			using difference_type = std::ptrdiff_t;
			using pointer = const FeatureVectorBinary**;
			using reference = const FeatureVectorBinary;

			Iterator(const DataView* view, int ix) : view(view), index(ix) {}

			const FeatureVectorBinary& operator*() const { return *(view->GetInstance(index)); }
			const FeatureVectorBinary* operator->() const { return view->GetInstance(index); }

			// Prefix increment
			Iterator& operator++() { index++; return *this; }

			// Postfix increment
			Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

			friend bool operator== (const Iterator& a, const Iterator& b) { return a.view == b.view && a.index == b.index; };
			friend bool operator!= (const Iterator& a, const Iterator& b) { return a.view != b.view || a.index != b.index; };

		private:
			const DataView* view;
			int index;
		};
		DataView() : data(nullptr), indices({}) {}
		DataView(const OriginalData* data, const std::vector<int>& indices) : data(data), indices(indices) {}
		inline int GetIndex(int ix) const { return indices.at(ix); }
		inline const std::vector<int>& GetIndices() const { return indices; }
		inline const FeatureVectorBinary* GetInstance(int ix) const { return data->at(indices.at(ix)); }
		inline int Size() const { return int(indices.size()); }
		void Add(int id) { indices.push_back(id); }

		inline Iterator begin() const { return Iterator(this, 0); }
		inline Iterator end() const { return Iterator(this, int(indices.size())); }
		inline Iterator cbegin() const { return Iterator(this, 0); }
		inline Iterator cend() const { return Iterator(this, int(indices.size())); }
	private:
		const OriginalData* data;
		std::vector<int> indices;
	};

	class BinaryData {
	public:
		BinaryData() : BinaryData(nullptr, {}, {}) {}
		BinaryData(const OriginalData* data) : BinaryData(data, {}, {}) {}
		BinaryData(const OriginalData* data, const std::vector<int> (&label_indices)[2], const std::vector<int> (&group_indices)[2]);
		BinaryData(const BinaryData& binary_data);
		~BinaryData() { } // Do not destroy data, since it might be shared with other objects. Leaf it to an explicit call to DestroyData

		inline int NumLabels() const { return 2; } // Only consider binary classification for the moment
		inline int NumGroups() const { return 2; } // Only consider binary discrimination for the moment
		inline int NumFeatures() const { return data->NumFeatures(); }
		inline int NumInstancesForLabel(int label) const { return label_view[label].Size(); }
		inline int NumInstancesForGroup(int group) const { return group_view[group].Size(); }
		inline int Size() const { return NumInstancesForLabel(0) + NumInstancesForLabel(1); }
		inline bool IsEmpty() const { return Size() == 0; }
		inline int TrivialSolution() const { return std::min(label_view[0].Size(), label_view[1].Size()); }

		inline void DestroyData() { delete data; }
		void SplitData(int feature, BinaryData& left, BinaryData& right) const;
		void TrainTestSplitData(double test_percentage, BinaryData& train, BinaryData& test) const;

		const int GetIndex(int index) const {
			return index < label_view[0].Size() ?
				label_view[0].GetIndex(index) :
				label_view[1].GetIndex(index - label_view[0].Size());
		}

		inline const int GetLabel(int index) const {
			return index < label_view[0].Size() ? 0 : 1;
		}

		const int GetGroup(int index) const {
			return index < label_view[0].Size() ?
				data->GetGroup(label_view[0].GetIndex(index)) :
				data->GetGroup(label_view[1].GetIndex(index - label_view[0].Size()));
		}

		const FeatureVectorBinary* GetInstance(int index) const {
			return index < label_view[0].Size() ?
				label_view[0].GetInstance(index) :
				label_view[1].GetInstance(index - label_view[0].Size());
		}
		inline const auto& GetInstancesForLabel(int label) const { return label_view[label]; }
		inline const auto& GetInstancesForGroup(int group) const { return group_view[group]; }
		int GetLabelGroupCount(int label, int group) const;

		void AddInstance(int id);

		inline double ComputeSparsity() const { return data->ComputeSparsity(); }
		void PrintStats() const;
		friend std::ostream& operator<<(std::ostream& os, const BinaryData& data);

		inline bool IsHashSet() const { return hash_value_ != -1; }
		inline long long GetHash() const {
			runtime_assert(IsHashSet())
				return hash_value_;
		}
		void SetHash(long long new_hash);

		inline bool IsClosureSet() const { return is_closure_set_; }
		inline const Branch& GetClosure() const {
			runtime_assert(IsClosureSet());
			return closure_;
		}
		void SetClosure(const Branch& closure);

		inline const OriginalData* GetOriginalData() const { return data; }
		inline const bool IsInitialized() const { return data != nullptr; }

		BinaryData* GetDeepCopy() const;

	private:
		long long hash_value_;

		bool is_closure_set_;
		Branch closure_;
		
		const OriginalData* data;
		DataView label_view[2];
		DataView group_view[2];		
	};

	struct DataSummary {
		DataSummary() : size(0), group0_size(0), group1_size(0), discrimination_factor(1.0) {}
		DataSummary(const BinaryData& data) :
			size(data.Size()),
			group0_size(data.NumInstancesForGroup(0)),
			group1_size(data.NumInstancesForGroup(1)),
			discrimination_factor(1.0 / data.NumInstancesForGroup(0) + 1.0 / data.NumInstancesForGroup(1)) {
		}
		int size;
		int group0_size;
		int group1_size;
		double discrimination_factor;
	};
}
