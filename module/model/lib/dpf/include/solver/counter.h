/**
Partly from Emir Demirovic "MurTree bi-objective"
https://bitbucket.org/EmirD/murtree-bi-objective
*/
#pragma once
#include "base.h"

namespace DPF {
	class Counter {
	public:
		Counter() : num_features(0), data2d(nullptr) {}
		Counter(int num_features);
		~Counter();

		int Positives(int index_row, int index_column) const;
		int Negatives(int index_row, int index_column) const;
		int& CountLabel(int label, int index_row, int index_column);

		void ResetToZeros();

		bool operator==(const Counter& reference)  const;

	private:
		int NumElements() const;
		int IndexSymmetricMatrix(int index_row, int index_column)  const;

		int* data2d;
		int num_features;
	};
}