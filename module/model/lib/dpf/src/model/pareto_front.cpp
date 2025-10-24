#include "model/pareto_front.h"

const static double NODE_COST = 1e-4;

namespace DPF {
    
    bool ParetoFront::Contains(const PartialSolution& sol, int num_nodes, const DataSummary& data_summary) const {
#if USE_PRUNE
        double partial = sol.GetInbalance(data_summary);
        auto it = unique_imbalance.find(int(partial / DISC_EPS));
        if(it == unique_imbalance.end()) return false;
        return (it->second <= sol.GetMisclassifications() + NODE_COST * num_nodes);
#else
        if (uniques.size() < 5) // Number based on speed benchmarking tests
            return std::find(uniques.begin(), uniques.end(), sol) != uniques.end(); // Do linear search
        return uniques.find(sol) != uniques.end();
#endif
    } 

    void ParetoFront::Insert(const ParetoFront& pf) {
        for (auto& p : pf.front) {
            Insert(p);
        }
    }

    void ParetoFront::Insert(const InternalNodeDescription& p, bool test_unique) {
        int partial = int(p.node_compare.partial_discrimination / DISC_EPS);
        if (Size() == 0) {
            front.push_back(p);
#if USE_PRUNE
            unique_imbalance[partial] = p.GetMisclassifications() + NODE_COST * p.NumNodes();
            if (POST_PARETO || !compare_all)
                unique_imbalance_reference[partial] = 0; // Store index of this solution for possible future overwrite
#else
            uniques.insert(p.GetPartialSolution());
#endif
            return;
        }

        //Test Unique
#if USE_PRUNE
        auto it = unique_imbalance.find(partial);
        const double cost = p.GetMisclassifications() + NODE_COST * p.NumNodes();
        if (it != unique_imbalance.end()) {
            if (it->second <= cost) return;
            it->second = cost;
            if (POST_PARETO || !compare_all) {
                auto it = unique_imbalance_reference.find(partial);
                if (it != unique_imbalance_reference.end()) {
                    front[it->second] = p; //overwrite solution
                }
                return;
            }
        } else {
            unique_imbalance[partial] = cost;
        }
#else
        if (uniques.find(p.GetPartialSolution()) != uniques.end()) return;
        uniques.insert(p.GetPartialSolution());
#endif

#if USE_PRUNE    
        if (!POST_PARETO) {
            if (compare_all || p.GetBestDiscrimination() > 0) {

                for (size_t i = 0; i < Size(); i++) {
                    if (dom(front[i], p)) return;
                }

                size_t old_size = front.size();
                front.erase(std::remove_if(front.begin(), front.end(),
                    [&p, this](const InternalNodeDescription& n) -> bool {
                    return dom(p, n);
                }), front.end());

        
                //If keeping track of references, and references are changed, then
                if (!compare_all && old_size > front.size()) {
                    // update references
                    unique_imbalance_reference.clear();
                    for (size_t i = 0; i < front.size(); i++) {
                        unique_imbalance_reference[int(front[i].node_compare.partial_discrimination / DISC_EPS)] = i;
                    }
                }
            }
        }
#endif

        front.push_back(p);
#if USE_PRUNE
        if (POST_PARETO || !compare_all)
            unique_imbalance_reference[partial] = front.size() - 1;
#endif

    }

    //https://codereview.stackexchange.com/questions/206686/removing-by-indices-several-elements-from-a-vector
    template <typename T, typename Iter>
    void RemoveIndicesFromVector(std::vector<T>& v, Iter begin, Iter end)
        // requires std::is_convertible_v<std::iterator_traits<Iter>::value_type, std::size_t>
    {
        assert(std::is_sorted(begin, end));
        auto rm_iter = begin;
        std::size_t current_index = 0;

        const auto pred = [&](const T&) {
            // any more to remove?
            if (rm_iter == end) { return false; }
            // is this one specified?
            if (*rm_iter == current_index++) { return ++rm_iter, true; }
            return false;
        };

        v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());
    }

    void ParetoFront::Filter(const DataSummary& data_summary) {
        if (Size() == 0) return;
#if USE_PRUNE
        if (POST_PARETO) {
            std::vector<int> remaining;
            bool one_zero = false;
            size_t best_misc_ix = 0;
            for (size_t i = 0; i < Size(); i++) {
                if (!SMART_DOM || front[i].GetBestDiscrimination() >= DISC_EPS || front[i].GetWorstDiscrimination() <= front[i].GetBestDiscrimination() + DISC_EPS) {
                    remaining.push_back(i);
                } else if (front[i].GetBestDiscrimination() < DISC_EPS) {
                    if (!one_zero ||
                        front[i].GetMisclassifications() + NODE_COST * this->front[i].NumNodes() < front[best_misc_ix].GetMisclassifications() + NODE_COST * this->front[best_misc_ix].NumNodes()) {
                        one_zero = true;
                        best_misc_ix = i;
                    }
                }
            }
            size_t index = 0;
            if (remaining.size() > 0) {
                if (one_zero)
                    remaining.push_back(best_misc_ix);


                std::sort(remaining.begin(), remaining.end(),
                    [this](const int i, const int j) -> bool {
                    return this->front[i].GetMisclassifications() + NODE_COST * this->front[i].NumNodes() < this->front[j].GetMisclassifications() + NODE_COST * this->front[j].NumNodes();
                });
                std::vector<bool> remove_mask(front.size(), false);
                while (index < remaining.size()) {
                    size_t new_index = 1;
                    std::vector<int> remove;
                    for (size_t i = 0; i < remaining.size(); i++) {
                        if (index == i) continue;
                        if (dom(front[remaining[index]], front[remaining[i]])) {
                            remove.push_back(i);
                            remove_mask[remaining[i]] = true;
                        } else if (i <= index) new_index++;
                    }
                    RemoveIndicesFromVector(remaining, remove.begin(), remove.end());
                    index = new_index;
                }
                //https://stackoverflow.com/questions/33494364/remove-vector-element-use-the-condition-in-vectorbool   
                front.erase(std::remove_if(front.begin(), front.end(), [&remove_mask, &front=this->front](auto const& i) { return remove_mask.at(&i - front.data()); }), front.end());
            }
        }
#endif
        UpdateBestBounds(data_summary);
    }

    void ParetoFront::UpdateBestBounds(const PartialSolution& p, const DataSummary& data_summary) {
        bounds.lower_bound = std::min(bounds.lower_bound, p.GetMisclassifications());
        bounds.budget.min_balance = std::min(bounds.budget.min_balance, p.GetInbalance(data_summary));
        bounds.budget.max_balance = std::max(bounds.budget.max_balance, p.GetInbalance(data_summary));
    }

    void ParetoFront::UpdateBestBounds(const DataSummary& data_summary) {
        bounds = BestBounds();
        for (size_t i = 0; i < Size(); i++) {
            UpdateBestBounds(front[i].GetPartialSolution(), data_summary);
        }
    }

    void ParetoFront::RemoveTempData() {
#if USE_PRUNE
        unique_imbalance.clear();
        unique_imbalance_reference.clear();
#else
        uniques.clear();
#endif
    }

    void ParetoFront::FilterOnUpperBound(int upper_bound) {
        front.erase(std::remove_if(front.begin(), front.end(),
            [upper_bound, this](const InternalNodeDescription& n) -> bool {
            return n.GetMisclassifications() >= upper_bound;
        }), front.end());
    }

    void ParetoFront::FilterOnDiscriminationBounds(const Branch& branch, const DataSummary& data_summary, double cut_off_value) {
#if USE_PRUNE
        front.erase(std::remove_if(front.begin(), front.end(),
            [&branch, &data_summary, cut_off_value, this](const InternalNodeDescription& n) -> bool {
            return n.GetPartialSolution().GetBestDiscrimination(branch, data_summary) > cut_off_value + DISC_EPS;
        }), front.end());
#endif
    }

    void ParetoFront::FilterOnNumberOfNodes(int num_nodes) {
        front.erase(std::remove_if(front.begin(), front.end(),
            [num_nodes, this](const InternalNodeDescription& n) -> bool {
            return n.NumNodes() != num_nodes;
        }), front.end());
    }

    void ParetoFront::FilterOnImbalance(double min, double max, const DataSummary& data_summary) {
#if USE_PRUNE        
        front.erase(std::remove_if(front.begin(), front.end(),
            [min, max, &data_summary, this](const InternalNodeDescription& n) -> bool {
            return n.GetPartialSolution().GetInbalance(data_summary) < min - DISC_EPS ||
                n.GetPartialSolution().GetInbalance(data_summary) > max + DISC_EPS;
        }), front.end());
#endif
    }

    void ParetoFront::SortByMisclassifications() {
        std::sort(front.begin(), front.end(),
            [](const InternalNodeDescription& n1, const InternalNodeDescription& n2) -> bool {
#if USE_PRUNE        
            return n1.GetMisclassifications() + NODE_COST*n1.NumNodes() < n2.GetMisclassifications() + NODE_COST * n2.NumNodes();
#else
            return n1.GetMisclassifications() < n2.GetMisclassifications();
#endif
        });
    }

    void ParetoFront::SortByInbalance() {
#if USE_PRUNE   
        std::sort(front.begin(), front.end(),
            [](const InternalNodeDescription& n1, const InternalNodeDescription& n2) -> bool {
            return n1.node_compare.partial_discrimination < n2.node_compare.partial_discrimination;
        });
#endif
    }

    size_t ParetoFront::LowerBoundByInbalance(const DataSummary& data_summary, double lower_bound) const {
#if USE_PRUNE        
        size_t l = 0;
        size_t u = Size() - 1;
        size_t m;
        while (l <= u) {
            m = (l + u) / 2;
            double inbalance = front[m].GetPartialSolution().GetInbalance(data_summary);
            if (inbalance >= lower_bound - DISC_EPS) { // If precisely the same do not increase l
                if (m == 0) break;
                u = m - 1;
            } else {
                l = m + 1;
            }
        }
        runtime_assert(l >= 0 && l <= Size());
        runtime_assert(u >= 0 && u <= Size());
        return l;
#else
        return 0;
#endif
    }

    size_t ParetoFront::UpperBoundByInbalance(const DataSummary& data_summary, double upper_bound) const {
#if USE_PRUNE
        size_t l = 0;
        size_t u = Size() - 1;
        size_t m;
        while (l <= u) {
            m = (l + u) / 2;
            double inbalance = front[m].GetPartialSolution().GetInbalance(data_summary);
            if (inbalance > upper_bound + DISC_EPS) { // If precisely the same do not increase u
                if (m == 0) return 0;
                u = m - 1;
            } else {
                l = m + 1;
            }
        }
        runtime_assert(l >= 0 && l <= Size());
        runtime_assert(u >= 0 && u <= Size());
        return l;
#else
        return Size();
#endif
    }

    bool ParetoFront::dom(const InternalNodeDescription& p1, const InternalNodeDescription& p2) const {
#if USE_PRUNE        
        return (p1.GetMisclassifications() + p1.NumNodes() * NODE_COST <= p2.GetMisclassifications() + p2.NumNodes() * NODE_COST
            && p1.GetWorstDiscrimination() <= p2.GetBestDiscrimination() + DISC_EPS);
#else
        return (p1.GetMisclassifications() <= p2.GetMisclassifications()
            && p1.GetWorstDiscrimination() <= p2.GetBestDiscrimination() - DISC_EPS);
#endif
    }
}