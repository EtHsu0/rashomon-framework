/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "utils/parameter_handler.h"
#include "utils/stopwatch.h"
#include "solver/solver.h"
#include "model/internal_node_description.h"

using namespace std;


int main(int argc, char* argv[]) {
	DPF::ParameterHandler parameters = DPF::ParameterHandler::DefineParameters();

	if (argc > 1) {
		parameters.ParseCommandLineArguments(argc, argv);
	} else {
		cout << "No paremeters specified." << endl << endl;
		parameters.PrintHelpSummary();
		exit(1);
	}

	if (parameters.GetBooleanParameter("verbose")) { parameters.PrintParameterValues(); }
	if (parameters.GetIntegerParameter("random-seed") == -1) { srand(time(0)); } else { srand(parameters.GetIntegerParameter("random-seed")); }


	parameters.CheckParameters();

	DPF::Stopwatch stopwatch;
	stopwatch.Initialise(0);

	if (parameters.GetBooleanParameter("verbose")) { std::cout << "Optimal tree computation started!\n"; }
	DPF::Solver dpf_solver(parameters);
	clock_t clock_before_solve = clock();
	DPF::SolverResult result;
	if (parameters.GetStringParameter("mode") == "hyper") {
		result = dpf_solver.HyperSolve();
	} else {
		result = dpf_solver.Solve();
	}
	std::cout << "TIME: " << stopwatch.TimeElapsedInSeconds() << " seconds\n";
	std::cout << "CLOCKS FOR SOLVE: " << ((double)clock() - (double)clock_before_solve) / CLOCKS_PER_SEC << "\n";

	if (parameters.GetBooleanParameter("verbose")) {
		if (result.IsFeasible()) {
			std::ofstream myfile;
			myfile.open(parameters.GetStringParameter("outfile"));
			result.PrintAllTree(myfile);
			myfile.close();
			if (parameters.GetStringParameter("mode") == "pareto") {
				const auto solutions = result.GetSolutionsInOrder();
				int count = 0;
				int best_misclassification_score = INT32_MAX;
				double disc_cut_off = parameters.GetFloatParameter("stat-test-value");
				for (auto& s : solutions) {
					std::cout << "Solution " << count++ << ": " << s.GetMisclassifications() << "," << s.node_compare.partial_discrimination << std::endl;
					if (s.GetMisclassifications() < best_misclassification_score &&
						s.GetBestDiscrimination() <= disc_cut_off) {
						best_misclassification_score = s.GetMisclassifications();
					}
				}
				
				std::cout << "Found " << solutions.size() << " solutions." << std::endl;

				if (result.IsProvenOptimal()) {
					auto& performance = result.GetPerformanceByMisclassificationScore(best_misclassification_score);

					std::cout << "Best solution Train score:  Misclassifications: " << std::setw(8) << performance.train_misclassifications
						<< ", Accuracy: " << std::setw(5) << performance.train_accuracy
						<< ", Discrimination: " << std::setw(8) << performance.train_discrimination << std::endl;
					if (parameters.GetFloatParameter("train-test-split") > 0) {
						std::cout << "Best solution Test  score:  Misclassifications: " << std::setw(8) << performance.test_misclassifications
							<< ", Accuracy: " << std::setw(5) << performance.test_accuracy
							<< ", Discrimination: " << std::setw(8) << performance.test_discrimination << std::endl;
					}
				}
			} else { // Only show best solution
				runtime_assert(result.performances.size() == 1);
				auto& performance = result.performances[0];
				std::cout << "Train score:  Misclassifications: " << std::setw(8) << performance.train_misclassifications
					<< ", Accuracy: " << std::setw(5) << performance.train_accuracy
					<< ", Discrimination: " << std::setw(8) << performance.train_discrimination << std::endl;
				if (parameters.GetFloatParameter("train-test-split") > 0) {
					std::cout << "Test  score:  Misclassifications: " << std::setw(8) << performance.test_misclassifications
						<< ", Accuracy: " << std::setw(5) << performance.test_accuracy
						<< ", Discrimination: " << std::setw(8) << performance.test_discrimination << std::endl;
				}
			}
						
		} else {
			std::cout << "No tree found\n";
		}
	}

	cout << endl << "DPF closed successfully!" << endl;
}

