/**
Partly from Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#pragma once
#include "base.h"
#include "model/container.h"
#include "model/binary_data.h"

namespace DPF {

	struct CacheEntry {
		CacheEntry(int depth, int num_nodes, int upper_bound) :
			depth(depth),
			num_nodes(num_nodes),
			upper_bound(upper_bound),
			bounds(0, DiscriminationBudget::nonRestrictedBudget) {
			runtime_assert(depth <= num_nodes);
		}

		CacheEntry(int depth, int num_nodes, int upper_bound, const DiscriminationBudget& budget, std::shared_ptr<AssignmentContainer> optimal_solutions, const BestBounds& bounds) :
			optimal_solutions(optimal_solutions),
			bounds(bounds),
			depth(depth),
			num_nodes(num_nodes),
			upper_bound(upper_bound),
			budget(budget) {
			runtime_assert(depth <= num_nodes);
		}

		std::shared_ptr<AssignmentContainer> GetOptimalSolution() const { 
			runtime_assert(IsOptimal());
			return std::make_shared<AssignmentContainer>(*(optimal_solutions.get()));
		}


		inline int GetLowerBound() const { runtime_assert(bounds.lower_bound >= 0); return bounds.lower_bound; }

		inline const DiscriminationBudget& GetBestBudget() const { runtime_assert(bounds.budget.IsRestricted()); return bounds.budget; }

		void SetOptimalSolutions(std::shared_ptr<AssignmentContainer> optimal_solutions) {
			runtime_assert(optimal_solutions.get() == nullptr || this->optimal_solutions->Size() == 0);
			this->optimal_solutions = optimal_solutions;
			if (optimal_solutions->Size() > 0) {
				bounds.lower_bound = optimal_solutions->GetLeastMisclassifications();
			}
		}

		void UpdateLowerBound(int lb) {
			runtime_assert(lb >= 0 && ((lb <= bounds.lower_bound && IsOptimal()) || !IsOptimal()));
			bounds.lower_bound = std::max(bounds.lower_bound, lb);
		}

		inline bool IsOptimal() const { return optimal_solutions.get() != nullptr && optimal_solutions->Size() > 0; }

		inline int GetNodeBudget() const { return num_nodes; }

		inline int GetDepthBudget() const { return depth; }
		
		inline int GetUpperBound() const { return upper_bound; }

		inline const DiscriminationBudget& GetDiscriminationBudget() const { return budget; }

	private:
		std::shared_ptr<AssignmentContainer> optimal_solutions;
		DiscriminationBudget budget;
		BestBounds bounds;
		int depth;
		int num_nodes;
		int upper_bound;
	};

	//key: a branch
	//value: cached value contains the optimal value and the lower bound
	class AbstractCache {
	public:

		virtual ~AbstractCache() {}

		//related to storing/retriving optimal branches
		virtual bool IsOptimalAssignmentCached(BinaryData&, const Branch& branch, int depth, int num_nodes, int upper_bound) = 0;
		virtual void StoreOptimalBranchAssignment(BinaryData&, const Branch& branch, std::shared_ptr<AssignmentContainer> optimal_solutions, int depth, int num_nodes, int upper_bound) = 0;
		virtual std::shared_ptr<AssignmentContainer> RetrieveOptimalAssignment(BinaryData&, const Branch& branch, int depth, int num_nodes, int upper_bound) = 0;
		virtual void TransferAssignmentsForEquivalentBranches(const BinaryData&, const Branch& branch_source, const BinaryData&, const Branch& branch_destination) = 0;//this updates branch_destination with all solutions from branch_source. Should only be done if the branches are equivalent.

		//related to storing/retrieving lower bounds
		virtual void UpdateLowerBound(BinaryData&, const Branch& branch, int lower_bound, int depth, int num_nodes) = 0;
		virtual int RetrieveLowerBound(BinaryData&, const Branch& branch, int depth, int num_nodes) = 0;

		//related to storing/retrieving discrimination budgets
		virtual void UpdateDiscrimationBudget(const Branch& org_branch, Branch& this_branch, BinaryData& data, const Branch& branch, int depth, int num_nodes, int upper_bound) = 0;
		virtual const DiscriminationBudget RetrieveBestBudgetBounds(BinaryData& data, const Branch& org_branch, const Branch& branch, int depth, int num_nodes, int upper_bound) = 0;

		//misc
		virtual int NumEntries() const = 0;

		virtual void DisableLowerBounding() = 0;
		virtual void DisableOptimalCaching() = 0;
	};
}