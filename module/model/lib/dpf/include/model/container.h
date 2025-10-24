#pragma once
#include "base.h"
#include "model/pareto_front.h"

namespace DPF {

	struct AssignmentContainer {

		AssignmentContainer(bool compare_all, int num_nodes) : num_nodes(num_nodes), solutions(compare_all), pruned(false) {}
		AssignmentContainer(const std::vector<InternalNodeDescription>& solutions, const BestBounds& bounds, int num_nodes) 
			: num_nodes(num_nodes), solutions(solutions, bounds), pruned(false) {}

		inline void Add(const InternalNodeDescription& node, bool test_unique=true) { solutions.Insert(node, test_unique); }
		inline void Add(const AssignmentContainer* container) { solutions.Insert(container->solutions); pruned |= container->pruned; }
		inline void FilterOnUpperbound(int upper_bound) { solutions.FilterOnUpperBound(upper_bound); }
		inline void FilterOnNumberOfNodes(int num_nodes) { solutions.FilterOnNumberOfNodes(num_nodes); }
		inline void FilterOnImbalance(double min, double max, const DataSummary& data_summary) { solutions.FilterOnImbalance(min, max, data_summary); }
		inline void FilterOnDiscriminationBounds(const Branch& branch, const DataSummary& data_summary, double cut_off_value) {
			solutions.FilterOnDiscriminationBounds(branch, data_summary, cut_off_value);
		}
		inline void SortByMisclassifications() { solutions.SortByMisclassifications(); }
		inline void SortByInbalance() { solutions.SortByInbalance(); }
		inline size_t LowerBoundByInbalance(const DataSummary& data_summary, double lower_bound) const { return solutions.LowerBoundByInbalance(data_summary, lower_bound); }
		inline size_t UpperBoundByInbalance(const DataSummary& data_summary, double upper_bound) const { return solutions.UpperBoundByInbalance(data_summary, upper_bound); }
		
		inline size_t Size() const { return solutions.Size(); }
		inline bool IsFeasible() const { return Size() > 0; }
		inline bool IsInfeasible() const { return Size() == 0; }
		inline int NumNodes() const { return IsFeasible() ? 0 : num_nodes; }
		inline const InternalNodeDescription& operator[](size_t i) const { return solutions[i]; }
		inline bool Contains(const PartialSolution& sol, int num_nodes, const DataSummary& data_summary) const 
			{ return solutions.Contains(sol, num_nodes, data_summary); }
		inline void RemoveTempData() { solutions.RemoveTempData(); }
		inline void SetPruned() { pruned = true; }
		inline bool GetPruned() const { return pruned; }
		inline void Filter(const DataSummary& data_summary) { solutions.Filter(data_summary); }
		
		void PrintSolutions() const {
			for (auto& node : solutions) {
				node.PrintSolution();
			}
		}

		inline const BestBounds& GetBestBounds() const { 
			runtime_assert(Size() == 0 || solutions.GetBestBounds().lower_bound < INT32_MAX);
			return solutions.GetBestBounds();
		}
		inline void UpdateBestBounds(const DataSummary& data_summary) { solutions.UpdateBestBounds(data_summary); }

		int GetLeastMisclassifications() const {
			runtime_assert(IsFeasible());
			return GetBestBounds().lower_bound;
		}

		double GetMinBalance(const DataSummary& data_summary) const {
			runtime_assert(IsFeasible());
			return GetBestBounds().budget.min_balance;
		}

		double GetMaxBalance(const DataSummary& data_summary) const {
			runtime_assert(IsFeasible());
			return GetBestBounds().budget.max_balance;
		}

		ParetoFront solutions;
		int num_nodes;
		bool pruned; // true if at least one solution is not added because of the disc. constraint
	};
}