"""mia_metric.py: Membership Inference Attack metrics for privacy evaluation.

This module implements membership inference attacks to assess privacy leakage from
machine learning models. Tests whether an attacker can determine if a specific
sample was part of the training set.
"""
import logging
from module.metric.base_metrics import BaseMetric, register_metric
from sklearn.metrics import accuracy_score
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.attacks.inference.membership_inference import (LabelOnlyDecisionBoundary,
                MembershipInferenceBlackBox, MembershipInferenceBlackBoxRuleBased)
from typing import Dict
import numpy as np


@register_metric()
class MiaMMetric(BaseMetric):
    """Membership Inference Attack metric for privacy evaluation.
    
    Evaluates privacy leakage by training attacks to distinguish between training
    set members and non-members. Always uses binary features when available to
    match the model's training data representation.
    """

    NAME = "membership_inference"
    REQUIRES_BINARY_FEATURES = True  # MIA always uses binary features when available
    
    def setup(self, model, hparams, split_data) -> None:
        """Prepare member and non-member sets for MIA evaluation.
        
        Creates disjoint sets of training members and test non-members for both
        training the attack model and evaluating attack success rate.
        
        Args:
            model: Trained target model.
            hparams (Hparams): Hyperparameters (unused but required).
            split_data (Split): Data split with train/test sets.
        """
        # Always use binary features if binarization was enabled during training
        # This ensures MIA operates on the same input space as the model
        if hasattr(split_data, 'binarizer') and split_data.binarizer is not None:
            binary_data = split_data.get_binary_data()
            X_train_preprocessed = binary_data['X_train']
            X_test_preprocessed = binary_data['X_test']
        else:
            # Fall back to continuous features if no binarization
            X_train_preprocessed = split_data.preprocess(split_data.X_train)
            X_test_preprocessed = split_data.preprocess(split_data.X_test)
        
        member_size = X_train_preprocessed.shape[0]
        non_member_size = X_test_preprocessed.shape[0]

        # MIA Setup: We need 4 disjoint sets for proper membership inference evaluation:
        # 1. Members (from train set) to train the MIA attack model
        # 2. Non-members (from test set) to train the MIA attack model
        # 3. Members (from train set) to evaluate attack success rate
        # 4. Non-members (from test set) to evaluate attack success rate
        #
        # Target: 500 samples per set (2000 total: 1000 for training attack, 1000 for evaluating attack)
        # Evaluation uses balanced sets: 500 members + 500 non-members = 1000 total samples
        
        # Calculate how many we can allocate from each population
        mia_train_size = min(500, member_size // 2)  # Members for training MIA
        mia_eval_size = min(500, member_size - mia_train_size)  # Members for eval
        
        mia_test_size = min(500, non_member_size // 2)  # Non-members for training MIA
        mia_eval_nonmember_size = min(500, non_member_size - mia_test_size)  # Non-members for eval
        
        # Ensure evaluation sets are balanced (same number of members and non-members)
        mia_eval_size = min(mia_eval_size, mia_eval_nonmember_size)
        
        # Log the final allocation for transparency
        logger = logging.getLogger("MIA Metric Setup")
        logger.info(f"MIA sample allocation - Train attack: {mia_train_size} members + {mia_test_size} non-members = {mia_train_size + mia_test_size}")
        logger.info(f"MIA sample allocation - Eval attack: {mia_eval_size} members + {mia_eval_size} non-members = {mia_eval_size * 2}")

        # Sample disjoint sets: first batch for training MIA, second batch for evaluating MIA
        member_idx = hparams.rng.choice(member_size, size=mia_train_size + mia_eval_size, replace=False)
        nonmember_idx = hparams.rng.choice(non_member_size, size=mia_test_size + mia_eval_size, replace=False)

        X_member, X_nonmember = X_train_preprocessed[member_idx], X_test_preprocessed[nonmember_idx]
        y_member, y_nonmember = split_data.y_train[member_idx], split_data.y_test[nonmember_idx]

        # Split into non-overlapping train and eval sets
        self.X_member_train, self.y_member_train = X_member[:mia_train_size], y_member[:mia_train_size]
        self.X_nonmember_train = X_nonmember[:mia_test_size]
        self.y_nonmember_train = y_nonmember[:mia_test_size]

        X_member_eval, y_member_eval = X_member[mia_train_size:mia_train_size + mia_eval_size], y_member[mia_train_size:mia_train_size + mia_eval_size]
        X_nonmember_eval, y_nonmember_eval = X_nonmember[mia_test_size:mia_test_size + mia_eval_size], y_nonmember[mia_test_size:mia_test_size + mia_eval_size]

        assert X_member_eval.shape[0] == X_nonmember_eval.shape[0] and \
                X_member_eval.shape[0] == mia_eval_size, \
                f"Size mismatch between {X_member_eval.shape[0]} \
                        {X_nonmember_eval.shape[0]} {mia_eval_size}"
        
        self.X_eval = np.concatenate((X_member_eval, X_nonmember_eval))
        self.y_eval = np.concatenate((y_member_eval, y_nonmember_eval))
        self.eval_label = np.concatenate((np.ones(mia_eval_size), np.zeros(mia_eval_size)))
        assert self.X_eval.shape[0] == self.y_eval.shape[0] and \
            self.X_eval.shape[0] == self.eval_label.shape[0], \
            f"Size mismatch between {self.X_eval.shape[0]} {self.y_eval.shape[0]} \
                    {self.eval_label.shape[0]}"


    def compute(self, predictions, split_data, **params) -> Dict[str, float]:
        ## Label-Only Decision Boundary Attack
        model = predictions["model"]
        self.art_classifier = ScikitlearnClassifier(model=model, clip_values=(0,1))
        logger = logging.getLogger("MIA Metric")
        logger.info("Running Label-Only Decision Boundary Attack (unsupervised)")
        attack = LabelOnlyDecisionBoundary(self.art_classifier)
        try:
            attack.calibrate_distance_threshold_unsupervised(top_t=50, num_samples=500,
                    max_queries=5,
                    batch_size=512, verbose=False)
            pred_label = attack.infer(self.X_eval, self.y_eval, verbose=False)
            label_only_unsupervised_accuracy = accuracy_score(self.eval_label, pred_label)
        except:
            label_only_unsupervised_accuracy = -1.0

        logger.info("Running Label-Only Decision Boundary Attack (supervised)")
        attack = LabelOnlyDecisionBoundary(self.art_classifier)
        try:
            attack.calibrate_distance_threshold(self.X_member_train, self.y_member_train,self.X_nonmember_train, self.y_nonmember_train, verbose=False)
            pred_label = attack.infer(self.X_eval, self.y_eval, verbose=False)
            label_only_supervised_accuracy = accuracy_score(self.eval_label, pred_label)
        except:
            label_only_supervised_accuracy = -1.0

        logger.info("Running Rule-Based Attacks")
        # Rule-Based Attack
        attack = MembershipInferenceBlackBoxRuleBased(self.art_classifier)
        pred_label = attack.infer(self.X_eval, self.y_eval)
        rule_based_accuracy = accuracy_score(self.eval_label, pred_label)
        # Black-Box Attack
        logger.info("Running Black-Box Attacks")
        attack = MembershipInferenceBlackBox(self.art_classifier)
        attack.fit(self.X_member_train, self.y_member_train, self.X_nonmember_train, self.y_nonmember_train, verbose=False)
        pred_label = attack.infer(self.X_eval, self.y_eval)
        blackbox_accuracy = accuracy_score(self.eval_label, pred_label)
        del self.art_classifier

        return {
            "mia_label_only_unsupervised_accuracy": label_only_unsupervised_accuracy,
            "mia_label_only_supervised_accuracy": label_only_supervised_accuracy,
            "mia_rule_based_accuracy": rule_based_accuracy,
            "mia_blackbox_accuracy": blackbox_accuracy
        }

    def cleanup(self):
        del self.X_member_train
        del self.y_member_train
        del self.X_nonmember_train
        del self.y_nonmember_train
        del self.X_eval
        del self.y_eval
        del self.eval_label
