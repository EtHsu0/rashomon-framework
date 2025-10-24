#include "utils/parameter_handler.h"
#include "solver/solver.h"
#include "model/internal_node_description.h"

using namespace std;

#define TIGHTNESS 0
#define LEAF_SIZE 0

int main(int argc, char* argv[]) {
	DPF::ParameterHandler parameters = DPF::ParameterHandler::DefineParameters();
	parameters.DefineNewCategory("Benchmark");
	parameters.DefineIntegerParameter("min-depth", "Minimum depth for benchmarking.", 3, "Benchmark", 0, 100);
	parameters.DefineIntegerParameter("repetitions", "Number of repetitions in benchmark", 1, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("min-num-features", "Minimum of features to consider in benchmarking", 1, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("feature-step", "Step size in enumerating different number of features in benchmarking", 1, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("min-num-instances", "Minimum number of instances to consider in benchmarking", 500, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("max-num-instances", "Maximum number of instances to consider in benchmarking", INT32_MAX, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("num-instances-step", "Step size in enumearting number of instances in benchmarking", 1, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("min-min-leaf-node-size", "Minimum min. leaf node size to consider in benchmarking", 1, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("max-min-leaf-node-size", "Maximum min. leaf node size to consider in benchmarking", INT32_MAX, "Benchmark", 1, INT32_MAX);
	parameters.DefineIntegerParameter("min-leaf-node-size-multiply-factor", "Maximum min. leaf node size to consider in benchmarking", 2, "Benchmark", 1, INT32_MAX);
	parameters.DefineFloatParameter("min-bias", "Minimum allowed bias to consider in benchmarking", 0.01, "Benchmark", DISC_EPS, 1);
	parameters.DefineFloatParameter("max-bias", "Maximum allowed bias to consider in benchmarking", 0.01, "Benchmark", DISC_EPS, 1);
	parameters.DefineFloatParameter("bias-step", "Step size in enumearting amount of allowed bias in benchmarking", DISC_EPS, "Benchmark", DISC_EPS, 1);

	if (argc > 1) parameters.ParseCommandLineArguments(argc, argv);
	if (parameters.GetBooleanParameter("verbose")) { parameters.PrintParameterValues(); }
	if (parameters.GetIntegerParameter("random-seed") == -1) { srand(time(0)); } else { srand(parameters.GetIntegerParameter("random-seed")); }
	parameters.SetIntegerParameter("max-num-nodes", (2 << (int(parameters.GetIntegerParameter("max-depth")) - 1)) - 1);

	parameters.CheckParameters();

	int min_features = int(parameters.GetIntegerParameter("min-num-features"));
	int max_features = int(parameters.GetIntegerParameter("max-num-features"));
	int feature_step = int(parameters.GetIntegerParameter("feature-step"));
	int min_instances = int(parameters.GetIntegerParameter("min-num-instances"));
	int max_instances = int(parameters.GetIntegerParameter("max-num-instances"));
	int instances_step = int(parameters.GetIntegerParameter("num-instances-step"));
	int min_min_leaf_size = int(parameters.GetIntegerParameter("min-min-leaf-node-size"));
	int max_min_leaf_size = int(parameters.GetIntegerParameter("max-min-leaf-node-size"));
	int min_leaf_size_multiply = int(parameters.GetIntegerParameter("min-leaf-node-size-multiply-factor"));
	double min_bias = parameters.GetFloatParameter("min-bias");
	double max_bias = parameters.GetFloatParameter("max-bias");
	double bias_step = parameters.GetFloatParameter("bias-step");
	int min_depth = int(parameters.GetIntegerParameter("min-depth"));
	int max_depth = int(parameters.GetIntegerParameter("max-depth"));
	double disc_cut_off = parameters.GetFloatParameter("stat-test-value");
	int repetitions = int(parameters.GetIntegerParameter("repetitions"));
	const bool test = parameters.GetFloatParameter("train-test-split") > 0;
	const bool output_all = true;
	bool instances_loop = max_instances < INT32_MAX;
	bool min_leaf_size_loop = max_min_leaf_size < INT32_MAX;
	bool max_bias_loop = std::abs(min_bias - max_bias) > DBL_EPSILON;
	std::cout << "D   " << "F    ";
	if (instances_loop) std::cout << "Instances ";
	if (max_bias_loop) std::cout << "Max-bias  ";
	if (min_leaf_size_loop) std::cout << "Leaf-Size ";
	std::cout << "Runtime   "
		<< "Term.-T.  "
		<< "Merge-T.  "
		<< "Misclas.  "
		<< "Accuracy  "
		<< "Discrim.  ";
	if (test) {
		std::cout << "Test-Mis. "
			<< "Test-Acc. "
			<< "Test-Dis. "
			<< "Test-Prf0  "
			<< "Test-Prf1  ";
	}
	std::cout << std::endl;
	for (int depth = min_depth; depth <= max_depth; depth++) {
		parameters.SetIntegerParameter("max-depth", depth);
		parameters.SetIntegerParameter("max-num-nodes", (2 << (depth - 1)) - 1);
		for (int instances = min_instances; instances < max_instances; instances += instances_step) {
			if (instances_loop) parameters.SetIntegerParameter("num-instances", instances);
			for (double bias = min_bias; bias < max_bias+DISC_EPS; bias += bias_step) {
				if(max_bias_loop) parameters.SetFloatParameter("stat-test-value", bias);
				for (int leaf_size = min_min_leaf_size; leaf_size < max_min_leaf_size; leaf_size *= min_leaf_size_multiply) {
					if (min_leaf_size_loop) parameters.SetIntegerParameter("min-leaf-node-size", leaf_size);
					for (int feature = min_features; feature <= max_features; feature += feature_step) {
						parameters.SetIntegerParameter("max-num-features", feature);
						double total_runtime = 0;
						double total_terminal_runtime = 0;
						double total_merge_runtime = 0;
						std::vector<DPF::Performance> performances;
						for (int i = 0; i < repetitions; i++) {
							DPF::Solver dpf_solver(parameters);
							clock_t clock_before_solve = clock();
							DPF::SolverResult dpf_solver_result;
							if(parameters.GetStringParameter("mode") == "hyper")
								dpf_solver_result = dpf_solver.HyperSolve();
							else
								dpf_solver_result = dpf_solver.Solve();
							double runtime = ((double)clock() - (double)clock_before_solve) / CLOCKS_PER_SEC;
							total_runtime += runtime;
							auto& stats = dpf_solver.GetStatistics();
							total_terminal_runtime += stats.time_in_terminal_node;
							total_merge_runtime += stats.time_merging;

							if (dpf_solver_result.IsProvenOptimal() && dpf_solver_result.IsFeasible()) {
								auto& performance = dpf_solver_result.performances[0];
								performances.push_back(performance);
								if (output_all) {
									std::cout << setw(2) << std::left << depth << "  " << setw(3) << feature << "  ";
									if (instances_loop) std::cout << setw(8) << instances << "  ";
									if (max_bias_loop) std::cout << setw(8) << bias << "  ";
									if (min_leaf_size_loop) std::cout << setw(8) << leaf_size << "  ";
									std::cout << setw(8) << runtime << "  "
										<< setw(8) << stats.time_in_terminal_node << "  "
										<< setw(8) << stats.time_merging << "  "
										<< setw(8) << performance.train_misclassifications << "  "
										<< setw(8) << performance.train_accuracy << "  "
										<< setw(8) << performance.train_discrimination << "  ";
									if (test) {
										std::cout
											<< setw(8) << performance.test_misclassifications << "  "
											<< setw(8) << performance.test_accuracy << "  "
											<< setw(8) << performance.test_discrimination << "  "
											<< setw(8) << performance.test_performance_group0 << "  "
											<< setw(8) << performance.test_performance_group1 << "  ";
									}
									std::cout << std::endl;
								}
							} else if (output_all) {
								std::cout << setw(2) << std::left << depth << "  " << setw(3) << feature << "  "
									<< setw(8) << runtime << "  "
									<< setw(8) << stats.time_in_terminal_node << "  "
									<< setw(8) << stats.time_merging << "  "
									<< "    -     "
									<< "    -     "
									<< "    -     ";
								if (test) {
									std::cout
										<< "    -     "
										<< "    -     "
										<< "    -     "
										<< "    -     "
										<< "    -     ";
								}
								std::cout << std::endl;
							}
							dpf_solver.Reset();
						}
						/*auto performance = DPF::Performance::GetAverage(performances);
						std::cout << setw(2) << std::left << depth << "  " << setw(3) << feature << "  "
							<< setw(8) << (total_runtime / repetitions) << "  "
							<< setw(8) << (total_terminal_runtime / repetitions) << "  "
							<< setw(8) << (total_merge_runtime / repetitions) << "  "
							<< setw(8) << performance.train_misclassifications << "  "
							<< setw(8) << performance.train_accuracy << "  "
							<< setw(8) << performance.train_discrimination << "  ";
						if (test) {
							std::cout
								<< setw(8) << performance.test_misclassifications << "  "
								<< setw(8) << performance.test_accuracy << "  "
								<< setw(8) << performance.test_discrimination << "  "
								<< setw(8) << performance.test_performance_group0 << "  "
								<< setw(8) << performance.test_performance_group1 << "  ";
						}
						std::cout << std::endl;*/
					}
					if (!min_leaf_size_loop) break;
				}
			}
			if (!instances_loop) break;
		} 

	}
}