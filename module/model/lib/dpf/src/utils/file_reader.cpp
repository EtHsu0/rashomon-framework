/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "utils/file_reader.h"

namespace DPF {
	BinaryData* FileReader::ReadDataDL(std::string filename, int num_instances, int max_num_features, int duplicate_instances_factor) {

		std::ifstream file(filename.c_str());

		if (!file) { std::cout << "Error: File " << filename << " does not exist!\n"; runtime_assert(file); }

		std::string line;
		int id = 0;
		int num_features = INT32_MAX;
		bool include_all = num_instances == INT32_MAX;
		int available_instances = INT32_MAX;
		std::vector<int> indices;
		if (!include_all) {
			available_instances = 0;
			while (std::getline(file, line)) {
				available_instances++;
			}
			file.clear();
			file.seekg(0);
			if (available_instances < num_instances) {
				include_all = true;
			} else {
				indices.resize(available_instances);
				std::iota(indices.begin(), indices.end(), 0);
				static auto rng = std::default_random_engine{};
				rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
				std::shuffle(indices.begin(), indices.end(), rng);
				indices.resize(num_instances);
				std::sort(indices.begin(), indices.end());
			}
		}

		std::vector<int> label_indices[2];
		std::vector<int> group_indices[2];

		std::vector<const FeatureVectorBinary*> feature_vectors;
		std::vector<int> labels;
		std::vector<int> groups;

		int line_no = -1;
		int indices_no = 0;
		while (std::getline(file, line)) {
			line_no++;
			runtime_assert(num_features == INT32_MAX || num_features == int((line.size() - 1) / 2 - 1));
			if (num_features == INT32_MAX) { num_features = int((line.size() - 1) / 2 - 1); }
			if (!include_all && line_no != indices[indices_no]) continue;
			indices_no++;

			std::istringstream iss(line);
			//the first value in the line is the label,
			//The second value in the line is the group
			// followed by 0-1 features
			int label;
			int group;
			iss >> label;
			runtime_assert(label == 0 || label == 1); // For now only accept binary classifications
			iss >> group;
			runtime_assert(group == 0 || group == 1); // For now only accept binary groups
			std::vector<bool> v;
			for (int i = 0; i < num_features; i++) {
				uint32_t temp;
				iss >> temp;
				runtime_assert(temp == 0 || temp == 1);
				if (i >= max_num_features) continue;
				v.push_back(temp);
			}

			for (int i = 0; i < duplicate_instances_factor; i++) {
				feature_vectors.push_back(new FeatureVectorBinary(v, id));
				labels.push_back(label);
				groups.push_back(group);
				label_indices[label].push_back(id);
				group_indices[group].push_back(id);
				id++;
				if (id >= num_instances) break;
			}
			if (id >= num_instances) break;
		}
		OriginalData* fvs = new OriginalData(feature_vectors, labels, groups);
		BinaryData* data = new BinaryData(fvs, label_indices, group_indices);

		return data;
	}
}