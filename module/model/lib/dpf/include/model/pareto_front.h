/**
Partly from Emir Demirovic "MurTree bi-objective"
https://bitbucket.org/EmirD/murtree-bi-objective
*/
#pragma once
#include "base.h"
#include "model/internal_node_description.h"


namespace DPF {

    struct BestBounds {
        BestBounds() : lower_bound(INT32_MAX), budget(1.0, -1.0) {}
        BestBounds(int lb, const DiscriminationBudget& budget) : lower_bound(lb), budget(budget) {}
        int lower_bound;
        DiscriminationBudget budget;
    };

    class ParetoFront {
    public:

        struct Iterator {
            using iterator_category = std::forward_iterator_tag;
            using value_type = InternalNodeDescription;
            using difference_type = std::ptrdiff_t; //Not sure about this one
            using pointer = InternalNodeDescription*;
            using reference = InternalNodeDescription&;

            Iterator(const ParetoFront* pf, size_t ix) : pf(pf), index(ix) {}

            inline const InternalNodeDescription& operator*() const { return (*pf)[index]; }
            inline const InternalNodeDescription* operator->() const { return &((*pf)[index]);
        }

            // Prefix increment
            inline Iterator& operator++() { index++; return *this; }

            // Postfix increment
            inline  Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

            inline friend bool operator== (const Iterator& a, const Iterator& b) { return &(a.pf) == &(b.pf) && a.index == b.index; };
            inline friend bool operator!= (const Iterator& a, const Iterator& b) { return a.pf != b.pf || a.index != b.index; };

        private:
            const ParetoFront* pf;
            size_t index;
        };

        ParetoFront() : compare_all(false || !SMART_DOM) {}
        ParetoFront(bool compare_all) : compare_all(compare_all) {}
        ParetoFront(const std::vector<InternalNodeDescription>& front, const BestBounds& bounds) : compare_all(false || !SMART_DOM), front(front), bounds(bounds) {}

        inline size_t Size() const { return front.size(); }
        inline const BestBounds& GetBestBounds() const { return bounds; }

        inline const InternalNodeDescription& operator[](size_t i) const { return front[i]; }
        inline const std::vector<InternalNodeDescription>& GetSolutions() const { return front; }

        void Insert(const ParetoFront& pf);
        void Insert(const InternalNodeDescription& p, bool test_unique = true);
        void UpdateBestBounds(const PartialSolution& p, const DataSummary& data_summary);
        void UpdateBestBounds(const DataSummary& data_summary);

        bool Contains(const PartialSolution& sol, int num_nodes, const DataSummary& data_summary) const;

        void RemoveTempData();
        void FilterOnUpperBound(int upper_bound);
        void FilterOnDiscriminationBounds(const Branch& branch, const DataSummary& data_summary, double cut_off_value);
        void FilterOnNumberOfNodes(int num_nodes);
        void FilterOnImbalance(double min, double max, const DataSummary& data_summary);
        void SortByMisclassifications();
        void SortByInbalance();
        size_t LowerBoundByInbalance(const DataSummary& data_summary, double lower_bound) const;
        size_t UpperBoundByInbalance(const DataSummary& data_summary, double upper_bound) const;
        void Filter(const DataSummary& data_summary);

        inline Iterator begin() const { return Iterator(this, 0); }
        inline Iterator end() const { return Iterator(this, front.size()); }
        inline Iterator cbegin() const { return Iterator(this, 0); }
        inline Iterator cend() const { return Iterator(this, front.size()); }

     private:

         inline bool eq(const InternalNodeDescription& p1, const InternalNodeDescription& p2) const {
            return (p1.GetPartialSolution() == p2.GetPartialSolution());
        }
         bool dom(const InternalNodeDescription& p1, const InternalNodeDescription& p2) const;

        std::vector<InternalNodeDescription> front;
#if USE_PRUNE
        std::unordered_map<int, double> unique_imbalance;
        std::unordered_map<int, int> unique_imbalance_reference;
#else
        std::unordered_set<PartialSolution> uniques;
#endif
        BestBounds bounds;
        bool compare_all;
    };
}

