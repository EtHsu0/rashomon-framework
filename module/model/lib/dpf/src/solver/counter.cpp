/**
Partly from Emir Demirovic "MurTree bi-objective"
https://bitbucket.org/EmirD/murtree-bi-objective
*/
#include "solver/counter.h"

namespace DPF {
	Counter::Counter(int num_features) :
		data2d(nullptr),
		num_features(num_features) {
		data2d = new int[NumElements()];
		ResetToZeros();
	}

	Counter::~Counter() {
		if(data2d != nullptr)
			delete[] data2d;
		data2d = nullptr;
	}

	int Counter::Positives(int index_row, int index_column)  const {
		runtime_assert(index_row <= index_column);
		int index = 2 * IndexSymmetricMatrix(index_row, index_column);
		return data2d[index];
	}

	int Counter::Negatives(int index_row, int index_column) const {
		runtime_assert(index_row <= index_column);
		int index = 1 + 2 * IndexSymmetricMatrix(index_row, index_column);
		return data2d[index];
	}

	int& Counter::CountLabel(int label, int index_row, int index_column) {
		runtime_assert(index_row <= index_column);
		int index = (1 - label) + 2 * IndexSymmetricMatrix(index_row, index_column);
		return data2d[index];
	}

	void Counter::ResetToZeros() {
		for (int i = 0; i < NumElements(); i++) { data2d[i] = 0; }
	}

	bool Counter::operator==(const Counter& reference)  const {
		if (num_features != reference.num_features) { return false; }

		for (int i = 0; i < NumElements(); i++) {
			if (data2d[i] != reference.data2d[i]) { return false; }
		}
		return true;
	}

	int Counter::NumElements() const {
		return num_features * (num_features + 1); //recall that the matrix is symmetric, and effectively each entry stores two integers (one for counting positives and one for counting negatives)
	}

	int Counter::IndexSymmetricMatrix(int index_row, int index_column)  const {
		runtime_assert(index_row <= index_column);
		return num_features * index_row + index_column - index_row * (index_row + 1) / 2;
	}
}