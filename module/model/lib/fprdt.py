import numpy as np
import json
from numba import jit
import time

_TREE_LEAF = -1
_TREE_UNDEFINED = -2

LEFT = 0
LEFT_INTERSECT = 1
RIGHT_INTERSECT = 2
RIGHT = 3


def convert_numpy(obj):
    """
    Convert numpy ints and floats to python types. Useful when converting objects to JSON.

    Parameters
    ----------
    obj : {np.int32, np.int64, np.float32, np.float64, np.longlong}
        Number to convert to python int or float.
    """
    if (
        isinstance(obj, np.int32)
        or isinstance(obj, np.int64)
        or isinstance(obj, np.longlong)
    ):
        return int(obj)
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Cannot convert type {type(obj)} to int or float")

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    else:
        return np.random.RandomState(seed)


class Node:
    """Base class for decision tree nodes, also functions as leaf."""

    def __init__(self, feature, left_child, right_child, value, parent = None):
        self.feature = feature
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent 
        self.value = value

    def predict(self, _):
        assert self.left_child == _TREE_LEAF
        assert self.right_child == _TREE_LEAF
        return self.value 
    
    def pretty_print(self, depth=0):
        indentation = depth * "  "
        return f"{indentation}return [{self.value[0]:.3f}, {self.value[1]:.3f}]"

    def to_json(self):
        return {
            "value": [self.value[0], self.value[1]],
        }

    def to_xgboost_json(self, node_id, depth, alpha):
        # Return leaf value in range [-1, 1]
        return {"nodeid": node_id, "leaf": alpha*(self.value[1] * 2 - 1)}, node_id

    def is_leaf(self):
        return self.left_child == _TREE_LEAF and self.right_child == _TREE_LEAF

    def prune(self, _):
        return self

class NumericalNode(Node):
    """
    Decision tree node for numerical decision (threshold).
    """
    def __init__(self, feature, threshold, left_child, right_child, value, parent=None):
        super().__init__(feature, left_child, right_child, value, parent)
        self.threshold = threshold


    def predict(self, sample):
        """
        Predict the class label of the given sample. Follow the left subtree
        if the sample's value is lower or equal to the threshold, else follow
        the right sub tree.
        """
        if self.left_child == _TREE_LEAF and self.right_child == _TREE_LEAF:
            return self.value

        comparison = sample[self.feature] <= self.threshold
        if comparison:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)

    def pretty_print(self, depth=0):
        indentation = depth * "  "
        return f"""{indentation}if x{self.feature} <= {self.threshold}:
                {self.left_child.pretty_print(depth + 1)}
                {indentation}else:
                {self.right_child.pretty_print(depth + 1)}"""

    def to_json(self):
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left_child": self.left_child.to_json(),
            "right_child": self.right_child.to_json(),
        }

    def to_xgboost_json(self, node_id, depth, alpha):
        left_id = node_id + 1
        if self.left_child == _TREE_LEAF:
            print("left child is leaf")
            print(type(self))
        left_dict, new_node_id = self.left_child.to_xgboost_json(left_id, depth + 1, alpha)

        right_id = new_node_id + 1
        right_dict, new_node_id = self.right_child.to_xgboost_json(right_id, depth + 1, alpha)

        return (
            {
                "nodeid": node_id,
                "depth": depth,
                "split": self.feature,
                "split_condition": self.threshold,
                "yes": left_id,
                "no": right_id,
                "missing": left_id,
                "children": [left_dict, right_dict],
            },
            new_node_id,
        )


NOGIL = True
@jit(nopython=True, nogil=NOGIL)
def _scan_numerical_feature_fast(
    samples,
    y,
    W,
    dec,
    inc,
    left_bound,
    right_bound,
):
    # TODO: so far we assume attack_mode is a tuple (dec, inc), and both
    # classes can move
    sort_order = samples.argsort()
    sorted_labels = y[sort_order]
    sorted_W = W[sort_order]
    sample_queue = samples[sort_order]
    dec_queue = sample_queue - dec
    inc_queue = sample_queue + inc

    # Initialize sample counters
    l_0 = l_1 = li_0 = li_1 = ri_0 = ri_1 = 0
    r_0 = W[y==0].sum()
    r_1 = W[y==1].sum()

    # Initialize queue values and indices
    sample_i = dec_i = inc_i = 0
    sample_val = sample_queue[0]
    dec_val = dec_queue[0]
    inc_val = inc_queue[0]

    least_loss = 10e9
    best_threshold = None
    best_values = None
    adv_loss = None
    
    while True:
        smallest_val = min(sample_val, dec_val, inc_val)

        # Find the current point and label from the queue with smallest value.
        # Also update the sample counters
        if sample_val == smallest_val:
            point = sample_val
            label = sorted_labels[sample_i]
            w = sorted_W[sample_i]

            if label == 0:
                ri_0 -= w
                li_0 += w
            else:
                ri_1 -= w
                li_1 += w

            # Update sample_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the sample_queue does not get picked
            if sample_i < sample_queue.shape[0] - 1:
                sample_i += 1
                sample_val = sample_queue[sample_i]
            else:
                sample_val = 10e9
        elif dec_val == smallest_val:
            point = dec_val
            label = sorted_labels[dec_i]
            w = sorted_W[dec_i]

            if label == 0:
                r_0 -= w
                ri_0 += w
            else:
                r_1 -= w
                ri_1 += w

            # Update dec_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the dec_queue does not get picked
            if dec_i < dec_queue.shape[0] - 1:
                dec_i += 1
                dec_val = dec_queue[dec_i]
            else:
                dec_val = 10e9
        else:
            point = inc_val
            label = sorted_labels[inc_i]
            w = sorted_W[inc_i]

            if label == 0:
                li_0 -= w
                l_0 += w
            else:
                li_1 -= w
                l_1 += w

            # Update inc_val and i to the values belonging to the next
            # sample in queue. If we reached the end of the queue then store
            # a high number to make sure the inc_queue does not get picked
            if inc_i < inc_queue.shape[0] - 1:
                inc_i += 1
                inc_val = inc_queue[inc_i]
            else:
                inc_val = 10e9

        if point >= right_bound:
            break

        # If the next point is not the same as this one
        next_point = min(sample_val, dec_val, inc_val)
        if next_point != point:
            # Calculate the k value according to the Appendix
            k1, k2, k3, k4, k5, k6 = l_0, l_1, r_0, r_1, li_0+ri_0, li_1+ri_1

            # Calculate the adversarial 0/1 loss
            losses = np.array([k2+k4+k6, k2+k3+k5+k6, k1+k4+k5+k6, k1+k3+k5])
            value_list = [(0,0),(0,1),(1,0),(1,1)]
            values = value_list[np.argmin(losses)]
            adv_loss = np.min(losses)

            # Maximize the margin of the split
            split = (point + next_point) * 0.5

            if (
                adv_loss is not None
                and (adv_loss < least_loss)
                and split > left_bound
                and split < right_bound
            ):
                least_loss = adv_loss
                best_threshold = split
                best_values = values
    return least_loss, best_threshold, best_values


class FPRDecisionTree():
    """
    A robust and fair decision tree for binary classification. Use class 0 for a negative and class 1 for a positive label.
    """
    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        epsilon=0,
        random_seed=None,
        if_record_loss = False
    ):
        """
        Parameters
        ----------
        max_depth : int, optional
            The maximum depth for the decision tree once fitted.
        min_samples_split : int, optional
            The minimum number of samples required to split a node.
        min_samples_leaf : int, optional
            The minimum number of samples required to make a leaf.
        max_features : int or {"sqrt", "log2"}, optional
            The number of features to consider while making each split, if None then all features are considered.
        epsilon : float, optional
            The value of the perturbation in L_\infinity attack.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.epsilon = epsilon
        self.if_record_loss = if_record_loss
        self.loss_op = []
        self.loss_01 = []
        self.n_categories_ = None
        self.random_seed = random_seed
        self.root_ = None
    
    def fit_BFS(self, X, y):
        """
        Build a robut binary decision tree from the trainin set with greedy
        splitting. All the candadiate nodes are in a queue. We search the queue
        until no node can be split.
        """
        self.y = y
        self.n_samples_, self.n_features_ = X.shape
        self.wrong_leaf_num = np.zeros([X.shape[0]], dtype=np.int32)
        self.random_state_ = check_random_state(self.random_seed)

        if self.max_features == "sqrt":
            self.max_features_ = int(np.sqrt(self.n_features_))
        elif self.max_features == "log2":
            self.max_features_ = int(np.log2(self.n_features_))
        elif self.max_features is None:
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = self.max_features
        if self.max_features_ == 0:
            self.max_features_ = 1

        if y.sum() <= y.shape[0]/2:
            self.wrong_leaf_num[y==1] += 1
            self.root_ = NumericalNode(None, None, _TREE_LEAF, _TREE_LEAF, (1,0))
        else:
            self.wrong_leaf_num[y==0] += 1 
            self.root_ = NumericalNode(None, None, _TREE_LEAF, _TREE_LEAF, (0,1))
        
        split_node_queue = [(self.root_, X, y, np.array([i for i in range(X.shape[0])]))]

        # We stop fitting the tree if no node can be split
        while True:
            for count1, (cur, tmpX, tmpy, tmpfall) in enumerate(split_node_queue):
                consider_index = []
                for count2, (i, label) in enumerate(zip(tmpfall, tmpy)):
                    if self.wrong_leaf_num[i] == 0:
                        consider_index.append(count2)
                    elif (self.wrong_leaf_num[i] == 1) and (label != cur.value[1]):
                        consider_index.append(count2)
                considerX = tmpX[consider_index]
                considery = tmpy[consider_index]

                if (considerX.shape[0]) < self.min_samples_split:
                    continue
                
                cur_loss = (considery!=cur.value[1]).sum()

                split_loss, feature, threshold, values = self.__best_adversarial_decision(
                    considerX, considery, np.ones(len(consider_index)), None
                )

                

                # We will split the node if the loss decreases
                if split_loss < cur_loss:
                    
                    X_left, y_left, X_right, y_right, \
                    fall_in_index_left, fall_in_index_right,\
                    _, _ \
                    = self.__split_left_right(tmpX, tmpy, tmpfall, None, threshold, feature)

                    if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                        continue

                    # print('split loss:', split_loss)
                    # print('cur loss:', cur_loss)

                    for i in tmpfall:
                        if self.y[i] != cur.value[1]:
                            self.wrong_leaf_num[i] -= 1
                    for i in fall_in_index_left:
                        if self.y[i] != values[0]:
                            self.wrong_leaf_num[i] += 1
                    for i in fall_in_index_right:
                        if self.y[i] != values[1]:
                            self.wrong_leaf_num[i] += 1

                    cur.feature = feature
                    cur.threshold = threshold
                    left_node = NumericalNode(None, None, _TREE_LEAF, _TREE_LEAF, (1-values[0],values[0]))
                    right_node = NumericalNode(None, None, _TREE_LEAF, _TREE_LEAF, (1-values[1],values[1]))
                    cur.left_child = left_node
                    cur.right_child = right_node
                    # print(count1)
                    # print(len(split_node_queue))
                    split_node_queue =  split_node_queue[count1+1:] + split_node_queue[:count1] + \
                            [(left_node, X_left, y_left, fall_in_index_left), (right_node, X_right, y_right, fall_in_index_right)]
                    # print(len(split_node_queue))
                    break
            else:
                break


        return self


    def fit(self, X, y, weights=None):
        """
        Build a robust and fair binary decision tree from the training set
        (X, y) using greedy splitting according to FPRDT from top to down.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.
        y : array-like of shape (n_samples,)
            The class labels as integers 0 (benign) or 1 (malicious)
        weights : array-like of shape (n_samples,)
            The weight of every singe sample.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.y = y
        self.n_samples_, self.n_features_ = X.shape
        self.loss_op = []
        self.loss_01 = [] 
        if weights is None:
            weights = np.ones(self.n_samples_)
        
        self.random_state_ = check_random_state(self.random_seed)
        
        # Record how many leaves that include a sample within perturbation 
        # and have a wrong prediction.  
        self.wrong_leaf_num = np.zeros([X.shape[0]], dtype=np.int32)
        
        # Initialize a root with negative prediction
        if y.sum() <= y.shape[0]/2:
            self.wrong_leaf_num[y==1] += 1
        # Initialize a root with positive prediction
        else:
            self.wrong_leaf_num[y==0] += 1
  
        # Initialize the number of the candidate feature 
        if self.max_features == "sqrt":
            self.max_features_ = int(np.sqrt(self.n_features_))
        elif self.max_features == "log2":
            self.max_features_ = int(np.log2(self.n_features_))
        elif self.max_features is None:
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = self.max_features
        if self.max_features_ == 0:
            self.max_features_ = 1

        # For each feature set the initial constraints for splitting
        constraints = []
        for feature_i in range(self.n_features_):
            constraints.append([np.min(X[:, feature_i]), np.max(X[:, feature_i])])

        self.loss_01.append(self.wrong_leaf_num.sum())
        self.loss_op.append(self.wrong_leaf_num.sum())

        self.root_ = self.__fit_recursive(X, y, np.array([i for i in range(X.shape[0])]), constraints, 0 if (y.sum()<=y.shape[0]/2) else 1, weights)

        return self

    def __fit_recursive(self, X, y, fall_in_index, constraints, prediction, weights, depth=0):
        """
        Recursively fit the decision tree on the training dataset.

        (X,y) is the samples which fall into the current node within perturbation.

        The fall_in_index is the index of samples which fall into the current node 
        within the perturbation.

        The constraints make sure that leaves are well formed, e.g. don't
        cross an earlier split. Stop when the depth has reached self.max_depth,
        when a leaf is pure or when the leaf contains too few samples.

        The prediction is the output of the node for samples which fall into this node.
        """
        if (
            depth == self.max_depth
            or np.sum(y == 0) == 0
            or np.sum(y == 1) == 0
        ):
            return self.__create_leaf(prediction)
        
        # We just consider the samples which:
        # 1. No leaves have wrong prediction for it.
        # 2. Or just the current node has a wrong prediction for it.
        consider_index = []
        for count, (i, label) in enumerate(zip(fall_in_index, y)):
            if self.wrong_leaf_num[i] == 0:
                consider_index.append(count)
            elif (self.wrong_leaf_num[i] == 1) and (label != prediction):
                consider_index.append(count)
        
        considerX = X[consider_index]
        considery = y[consider_index]
        considerW = weights[consider_index]

        if (considerX.shape[0]) < self.min_samples_split:
            return self.__create_leaf(prediction)

        # If we split this node, compute the best way.
        split_loss, feature, threshold, values = self.__best_adversarial_decision(
            considerX, considery, considerW, constraints
        )
        
        if (values is None) or (values[0] == values[1]):
            return self.__create_leaf(prediction)

        # If we do not split this node, calculate the loss.
        # current_loss = (considery!=prediction).sum()
        current_loss = considerW[considery!=prediction].sum()

        # diff_loss > 0 if split is better
        diff_loss = current_loss - split_loss

        if feature is None or diff_loss <= 0:
            return self.__create_leaf(prediction)

        # Assert that the split obeys constraints made by previous splits
        assert threshold >= constraints[feature][0]
        assert threshold < constraints[feature][1]
        
        # Split the node
        X_left, y_left, X_right, y_right, \
        fall_in_index_left, fall_in_index_right,\
        weights_left, weights_right \
        = self.__split_left_right(X, y, fall_in_index, weights, threshold, feature)

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return self.__create_leaf(prediction)
        
        for i in fall_in_index:
            if self.y[i] != prediction:
                self.wrong_leaf_num[i] -= 1
        for i in fall_in_index_left:
            if self.y[i] != values[0]:
                self.wrong_leaf_num[i] += 1
        for i in fall_in_index_right:
            if self.y[i] != values[1]:
                self.wrong_leaf_num[i] += 1

        # Record the loss
        if self.if_record_loss:            
            self.loss_01.append((self.wrong_leaf_num>=1).sum())
            self.loss_op.append(self.loss_op[-1]-current_loss+split_loss)

        # Set the right bound and store old one for after recursion
        old_right_bound = constraints[feature][1]
        constraints[feature][1] = threshold

        left_node = self.__fit_recursive(X_left, y_left, \
            fall_in_index_left, constraints, values[0], weights_left, depth + 1)

        # Reset right bound, set left bound, store old one for after recursion
        constraints[feature][1] = old_right_bound
        old_left_bound = constraints[feature][0]
        constraints[feature][0] = threshold
       
        right_node = self.__fit_recursive(X_right, y_right, \
            fall_in_index_right, constraints, values[1], weights_right, depth + 1)

        # Reset the left bound
        constraints[feature][0] = old_left_bound
        node = NumericalNode(feature, threshold, left_node, right_node, _TREE_UNDEFINED)

        return node

    def __create_leaf(self, value):
        """
        Create a leaf object that with value as the prediction.
        """
        return Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, (1-value, value))


    def __split_left_right(self, X, y, fall_in_index, weights, threshold, feature):
        index1 = []
        index2 = []
        l,r = self.epsilon, self.epsilon
        for count, x in enumerate(X):
            if x[feature] <= threshold+r:
                index1.append(count)
            if x[feature] > threshold-l:
                index2.append(count)
        if weights is None:
            return X[index1], y[index1], X[index2], y[index2], fall_in_index[index1], \
                fall_in_index[index2], None, None
        else:
            return X[index1], y[index1], X[index2], y[index2], fall_in_index[index1], \
                fall_in_index[index2], weights[index1], weights[index2]
        

    def __best_adversarial_decision(self, X, y, W, constraints):
        """
        Find the best split by iterating through each feature and scanning
        it for that feature's optimal split.
        """
        least_loss = 10e9
        best_feature = None
        best_threshold = None
        best_values = None

        # If there is a limit on features to consider in a split then choose
        # that number of random features.
        all_features = np.arange(self.n_features_)

        features = self.random_state_.choice(
            all_features, size=self.max_features_, replace=False
        )

        for feature in features:
            cur_loss, cur_threshold, cur_values = self.__scan_feature(
                X, y, W, feature, constraints
            )

            if cur_threshold is not None and cur_loss < least_loss:
                least_loss = cur_loss
                best_feature = feature
                best_threshold = cur_threshold
                best_values = cur_values

        return least_loss, best_feature, best_threshold, best_values

    def __scan_feature(self, X, y, W, feature, constraints):
        samples = X[:, feature]
        if constraints is None:
            constraint = [0,1]
        else:
            constraint = constraints[feature]
        return _scan_numerical_feature_fast(
            samples, y, W, self.epsilon, self.epsilon, *constraint
        )
    
    def predict_proba(self, X):
        """
        Predict the class of the input samples X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        class : array of shape (n_samples,)
            The class for each input sample of being malicious.
        """
        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

    def predict(self, X):
        """
        Predict the classes of the input samples X.

        The predicted class is the most frequently occuring class label in a
        leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """
        return np.round(self.predict_proba(X)[:, 1])
    
    def score(self, X, y):
        """
        Compute the accuracy of the classifier on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples,)
            The true class labels.

        Returns
        -------
        score : float
            The accuracy of the classifier.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def adversarial_attack_recur(self, node, X, y, fall_in_index, attacked):
        if node.is_leaf():
            for (label, i) in zip(y, fall_in_index):
                if label != node.value[1]:
                    attacked[i] = True
        else:
            index1 = []
            index2 = []
            l,r = self.epsilon, self.epsilon
            feature, threshold = node.feature, node.threshold
            for count, x in enumerate(X):
                if x[feature] <= threshold+r:
                    index1.append(count)
                if x[feature] > threshold-l:
                    index2.append(count)
            self.adversarial_attack_recur(node.left_child, X[index1], y[index1], \
                fall_in_index[index1], attacked)
            self.adversarial_attack_recur(node.right_child, X[index2], y[index2], \
                fall_in_index[index2], attacked)


    def adversarial_attack(self, X, y):
        """
        Compute whether we can attack the input dataser successfuly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        ----------
        attacked: array-like of shape (n_samples,)
                Indicate whether the corresponding sample can be attacked successfully.
        """
        attacked = [False] * X.shape[0]
        self.adversarial_attack_recur(self.root_, X, y, np.array([i for i in range(X.shape[0])]), attacked)
        return attacked


    def to_string(self):
        result = ""
        # result += f"Parameters: {self.get_params()}\n"

        if hasattr(self, "root_"):
            result += f"Tree:\n{self.root_.pretty_print()}"
        else:
            result += "Tree has not yet been fitted"

        return result

    def to_json(self, output_file="tree.json"):
        dictionary = {
            "params": self.get_params(),
        }
        if hasattr(self, "root_"):
            dictionary["tree"] = self.root_.to_json()
        else:
            dictionary["tree"] = None

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                json.dump(dictionary, fp, indent=2, default=convert_numpy)

    def to_xgboost_json(self, alpha=1, output_file=None):
        if hasattr(self, "root_"):
            dictionary, _ = self.root_.to_xgboost_json(0, 0, alpha)
        else:
            raise Exception("Tree is not yet fitted")

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                # If saving to file then surround dict in list brackets
                json.dump([dictionary], fp, indent=2, default=convert_numpy)
    
    def get_depth(self):
        return self.get_depth_recursive(self.root_, 0)
    
    def get_depth_recursive(self, node, depth):
        if node.is_leaf():
            return depth
        else:
            return max(self.get_depth_recursive(node.left_child, depth+1),self.get_depth_recursive(node.right_child, depth+1))
    
    def get_leaf_number(self):
        return self.get_leaf_number_recursive(self.root_,0)

    def get_leaf_number_recursive(self, node, leaf_num):
        if node.is_leaf():
            return 1
        else:
            return self.get_leaf_number_recursive(node.left_child, leaf_num) + self.get_leaf_number_recursive(node.right_child, leaf_num)
    
    def show_recur(self, node):
        #time.sleep(0.1)
        if node.is_leaf():
            print(node.value[1], end='')
            return 3
        sen = '%d/%.3f---'%(node.feature, node.threshold)
        print(sen, end='')
        height1 = self.show_recur(node.left_child)
        print('\033[%dD'%(len(sen)+1), end = '')

        print('\033[1B', end='')
        for _ in range(height1-1):
            print('|', end = '')
            print('\033[1B', end='')
            print('\033[1D', end='')
        print('-'*len(sen), end='')
        height2 = self.show_recur(node.right_child)
        print('\033[%dD'%(len(sen)), end='')
        print('\033[%dA'%(height1), end='')

        return height1+height2

    def show(self):
        leaf_num = self.get_leaf_number()
        print('\n'*(3*leaf_num+3), end='')
        print('\033[%dA'%(3*leaf_num+3), end='')
        h = self.show_recur(self.root_)
        for _ in range(3*leaf_num+3):
            print('')
    

    def get_feature_recur(self, node):
        if node.is_leaf():
            return []
        else:
            return [node.feature] + self.get_feature_recur(node.left_child) + self.get_feature_recur(node.right_child)

    def get_feature(self):
        return self.get_feature_recur(self.root_)
