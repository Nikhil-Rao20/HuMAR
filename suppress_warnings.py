"""
Warning suppression and compatibility fixes for training script.
This module suppresses all unnecessary warnings that clutter the training output.
"""

import warnings
import os
import logging

def suppress_all_warnings():
    """
    Suppress all warnings that clutter training output.
    """
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Suppress specific warning categories
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
    
    # Suppress matplotlib warnings
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    # Suppress PIL warnings
    logging.getLogger('PIL').setLevel(logging.ERROR)
    
    # Suppress numba warnings
    try:
        from numba.core.errors import NumbaWarning, NumbaDeprecationWarning
        warnings.filterwarnings('ignore', category=NumbaWarning)
        warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)
    except ImportError:
        pass
    
    # Suppress sklearn warnings
    try:
        import sklearn
        sklearn.set_config(print_changed_only=True)
    except:
        pass
    
    # Suppress pandas warnings
    try:
        import pandas as pd
        pd.options.mode.chained_assignment = None
    except:
        pass
    
    # Suppress PyArrow warnings
    try:
        import pyarrow as pa
        pa.set_options(warn_on_size_estimation=False)
    except:
        pass

if __name__ == "__main__":
    suppress_all_warnings()
    print("✅ All warnings suppressed")
