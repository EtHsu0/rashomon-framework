/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once

#include "base.h"
#include "solver/abstract_cache.h"
#include "solver/difference_computer.h"
#include "model/binary_data.h"
#include "model/branch.h"

namespace DPF {
	struct PairLowerBoundOptimal {
		PairLowerBoundOptimal(int lb, bool opt) :lower_bound(lb), optimal(opt) {}
		int lower_bound;
		bool optimal;
	};


	class SimilarityLowerBoundComputer {
	public:
		SimilarityLowerBoundComputer(int max_depth, int size, int num_instances);

		//computes the lower bound with respect to the data currently present in the data structure
		//note that this does not add any information internally
		//use 'UpdateArchive' to add datasets to the data structure
		PairLowerBoundOptimal ComputeLowerBound(BinaryData& data, Branch& branch, int depth, int size, int upper_bound, AbstractCache* cache);

		void UpdateDiscrimationBudget(const Branch& org_branch, Branch& this_branch, BinaryData& data, const Branch& branch, const DataSummary& data_summary, int depth, int size, int upper_bound, AbstractCache* cache);

		//adds the data, possibly replacing some previously stored dataset in case there the data structure is full. 
		//when replacing, it will find the most similar dataset and replace it with the input
		//TODO make it should replace the most disimilar dataset, and not the most similar?
		void UpdateArchive(BinaryData& data, Branch& branch, int depth);

		void Disable();

	private:

		struct ArchiveEntry {
			ArchiveEntry(BinaryData& d, Branch& b) :
				data(d),
				branch(b) {
			}

			BinaryData data;
			Branch branch;
		};

		void Initialise(int max_depth, int size);
		ArchiveEntry& GetMostSimilarStoredData(BinaryData& data, int depth);

		std::vector<std::vector<ArchiveEntry> > archive_;//archive_[depth][i]
		bool disabled_;
	};
}