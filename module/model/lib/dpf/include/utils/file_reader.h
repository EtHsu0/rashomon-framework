#pragma once
#include "base.h"

#include "model/feature_vector_binary.h"
#include "model/binary_data.h"

namespace DPF {
	class FileReader {
	public:
		//the DL file format has a instance in each row, where the label at the first position followed by the values of the binary features separated by whitespace
		//the method reads a file in the DL file format and a BinaryData object
		// duplicate_instances_factor states how many times each instance should be duplicated: this is only useful for testing/benchmarking purposes
		static BinaryData* ReadDataDL(std::string filename, int num_instances, int max_num_features, int duplicate_instances_factor = 1);

	};
}