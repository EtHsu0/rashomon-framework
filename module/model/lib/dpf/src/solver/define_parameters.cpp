/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#include "utils/parameter_handler.h"

namespace DPF {
	
	ParameterHandler ParameterHandler::DefineParameters() {
		ParameterHandler parameters;

		parameters.DefineNewCategory("Main Parameters");
		parameters.DefineNewCategory("Algorithmic Parameters");

		parameters.DefineStringParameter
		(
			"file",
			"Location to the dataset.",
			"", //default value
			"Main Parameters"
		);

		parameters.DefineFloatParameter
		(
			"time",
			"Maximum runtime given in seconds.",
			3600, //default value
			"Main Parameters",
			0, //min value
			INT32_MAX //max value
		);

		parameters.DefineStringParameter
		(
			"mode",
			"Mode for the algorithm (find best or find the pareto front)",
			"best", //default value
			"Main Parameters",
			{ "best", "pareto", "hyper"}
		);

		parameters.DefineStringParameter
		(
			"outfile",
			"Output file path",
			"example.json", //default value
			"Main Parameters"
		);

		parameters.DefineStringParameter
		(
			"trainfile",
			"Train file path",
			"",
			"Main Parameters"
		)

		parameters.DefineStringParameter
		(
			"testfile",
			"Test file path",
			"",
			"Main Parameters"
		);

		parameters.DefineIntegerParameter
		(
			"max-depth",
			"Maximum allowed depth of the tree, where the depth is defined as the largest number of *decision/feature nodes* from the root to any leaf. Depth greater than four is usually time consuming.",
			3, //default value
			"Main Parameters",
			0, //min value
			20 //max value
		);
		
		parameters.DefineIntegerParameter
		(
			"max-num-nodes",
			"Maximum number of *decision/feature nodes* allowed. Note that a tree with k feature nodes has k+1 leaf nodes.",
			7, //default value
			"Main Parameters",
			0,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"max-num-features",
			"Maximum number of features that are considered from the dataset (in order of appearance).",
			INT32_MAX, // default value,
			"Main Parameters",
			1,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"num-instances",
			"Number of instances that are considered from the dataset (in order of appearance).",
			INT32_MAX, // default value,
			"Main Parameters",
			1,
			INT32_MAX
		);

		parameters.DefineBooleanParameter
		(
			"verbose",
			"Determines if the solver should print logging information to the standard output.",
			true,
			"Main Parameters"
		);

		parameters.DefineBooleanParameter
		(
			"all-trees",
			"Instructs the algorithm to compute trees using all allowed combinations of max-depth and max-num-nodes. Used to stress-test the algorithm.",
			false,
			"Main Parameters"
		);

		parameters.DefineFloatParameter
		(
			"stat-test-value",
			"Cut-off value for the discrimination constraint (%)",
			0.01, //default value
			"Main Parameters",
			0, //min value
			1.0 //max value
		);

		parameters.DefineFloatParameter
		(
			"sparsity",
			"Sparsity parameter (minimum gain in accuracy (%) to justify adding an extra node)",
			1e-6, //default value
			"Main Parameters",
			0, //min value
			1.0 //max value
		);

		parameters.DefineFloatParameter
		(
			"train-test-split",
			"The percentage of instances for the test set",
			0.0, //default value
			"Main Parameters",
			0, //min value
			1.0 //max value
		);

		parameters.DefineIntegerParameter
		(
			"min-leaf-node-size",
			"The minimum size of leaf nodes",
			1, // default value
			"Main Parameters",
			1, //min value
			INT32_MAX // max value
		);

		parameters.DefineBooleanParameter
		(
			"similarity-lower-bound",
			"Activate similarity-based lower bounding. Disabling this option may be better for some benchmarks, but on most of our tested datasets keeping this on was beneficial.",
			true,
			"Algorithmic Parameters"
		);

		parameters.DefineStringParameter
		(
			"feature-ordering",
			"Feature ordering strategy used to determine the order in which features will be inspected in each node.",
			"in-order", //default value
			"Algorithmic Parameters",
			{ "in-order", "random", "gini" }
		);

		parameters.DefineIntegerParameter
		(
			"random-seed",
			"Random seed used only if the feature-ordering is set to random. A seed of -1 assings the seed based on the current time.",
			3,
			"Algorithmic Parameters",
			-1,
			INT32_MAX
		);

		parameters.DefineStringParameter
		(
			"cache-type",
			"Cache type used to store computed subtrees. \"Dataset\" is more powerful than \"branch\" but may required more computational time. Need to be determined experimentally. \"Closure\" is experimental and typically slower than other options.",
			"branch", //default value
			"Algorithmic Parameters",
			//{ "branch", "dataset", "closure" }
			{ "branch" }
		);

		parameters.DefineIntegerParameter
		(
			"duplicate-factor",
			"Duplicates the instances the given amount of times. Used for stress-testing the algorithm, not a practical parameter.",
			1,
			"Algorithmic Parameters",
			1,
			INT32_MAX
		);

		parameters.DefineIntegerParameter
		(
			"upper-bound",
			"Initial upper bound.",
			INT32_MAX, //default value
			"Algorithmic Parameters",
			0,
			INT32_MAX
		);

		return parameters;
	}
}