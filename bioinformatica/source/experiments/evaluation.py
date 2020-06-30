import numpy as np
from bioinformatica.source.commons import *
import scipy.stats
from bioinformatica.source.models.builder import Model


def test_models(models: List, statistical_test: scipy.stats, metric: Tuple[Callable, str], alpha: float) \
        -> Tuple[Model, float, str, float]:
    best_model, best_score = models[0], models[0].get_scores().get(metric[0].__name__)
    for current_model in models[1:]:
        comparison_scores = current_model.get_scores().get(metric[0].__name__)
        stats, p_value = statistical_test(best_score, comparison_scores)
        if p_value < alpha:
            if np.array(comparison_scores).mean() > np.array(best_score).mean():
                best_model, best_score = current_model, comparison_scores
    return best_model, alpha, metric[0].__name__, np.array(best_score).mean()