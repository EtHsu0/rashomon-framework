"""utils.py: General utility functions for experiments and analysis.

This module provides utility functions and classes for data management, JSON encoding,
and stability distribution generation used throughout the experiment framework.
"""
import json
import os
import argparse
import numpy as np


def save_data_as_dpf_csv(filename, X, y, sens):
    """Save dataset in DPF-compatible CSV format.
    
    Args:
        filename (str): Output filename (without .csv extension).
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        sens (np.ndarray): Sensitive attributes.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    y_col = np.ravel(y).reshape(-1, 1)
    sens = np.ravel(sens).reshape(-1, 1)
    data = np.hstack((y_col, sens, X))
    np.savetxt(f"{filename}.csv", data, delimiter=" ", fmt="%.6g")


def generate_stability_distribution(rng, mean, std, size):
    """Generate clipped normal distribution for stability experiments.
    
    Generates values from a normal distribution and clips them to the valid
    range (0, 1] for use with geometric distributions in stability testing.
    
    Args:
        rng (np.random.Generator): Random number generator.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        size (int): Number of samples to generate.
        
    Returns:
        np.ndarray: Clipped values in the range (0, 1].
    """
    values = rng.normal(loc=mean, scale=std, size=size)
    # Clip to ensure values are in (0, 1] range for geometric distribution
    return np.clip(values, np.nextafter(0.0, 1.0), 1.0)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling numpy data types.
    
    Converts numpy arrays, integers, and floats to
    JSON-serializable Python types. Large arrays are summarized rather
    than fully serialized.
    """
    
    def default(self, o):
        """Encode numpy objects to JSON-compatible types.
        
        Args:
            o: Object to encode.
            
        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(o, np.ndarray) or isinstance(o, list): 
            # Print if not large array:
            if isinstance(o, np.ndarray) and o.size < 50:
                return o.tolist()
            return f"ARRAY: {o.shape} {o.dtype}"
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return super().default(o)

