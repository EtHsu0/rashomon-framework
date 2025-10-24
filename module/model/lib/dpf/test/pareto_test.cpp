#include <stdio.h>
#include <stdlib.h>
#include "model/pareto_front.h"
#include "model/internal_node_description.h"

#define test_assert(x) {if (!(x)) { printf(#x); assert(#x); exit(1); }}

using namespace DPF;

InternalNodeDescription GetNode(int misclassifications, double best_discrimination, double worst_discrimination, const DataSummary& data_summary) {
	PartialSolution solution(misclassifications, 0, 0);
	InternalNodeDescription node(0, 0, solution, 1, 1, Branch(), data_summary);
	node.node_compare.best_discrimination = best_discrimination;
	node.node_compare.worst_discrimination = worst_discrimination;
	return node;
}

int main(int argc, char **argv) {
	ParetoFront front (true);
	DataSummary data_summary;
	data_summary.size = 100;
	data_summary.group0_size = 40;
	data_summary.group1_size = 60;
	data_summary.discrimination_factor = 1.0;

	front.Insert(GetNode(10, 2, 10, data_summary));
	front.Insert(GetNode(11, 11, 12, data_summary)); // This node should be dominated
	test_assert(front.Size() == 1);

	front.Insert(GetNode(8, 4, 6, data_summary));
	test_assert(front.Size() == 2 ); // This node should not be dominated

	// Insert point that dominates all
	front.Insert(GetNode(0, 0, 0, data_summary));
	test_assert(front.Size() == 1);
	test_assert(front[0].GetMisclassifications() == 0);
}