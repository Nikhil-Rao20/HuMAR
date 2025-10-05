from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_refcoco_config

# dataset loading
from .data.dataset_mappers.refcoco_mapper import RefCOCOMapper

# models
from .GRES import GRES
from .multitask_gres import MultitaskGRES  # Import multitask model to register it

# evaluation
from .evaluation.refer_evaluation import ReferEvaluator
