import numpy as np


def test_models(models, statistical_test, metric, alpha):
    best_model, best_score = models[0], models[0].scores.get(metric[0].__name__)
    for current_model in models[1:]:
        comparison_scores = current_model.scores.get(metric[0].__name__)
        stats, p_value = statistical_test(best_score, comparison_scores)
        if p_value < alpha:
            if np.array(comparison_scores).mean() > np.array(best_score).mean():
                best_model, best_score = current_model, comparison_scores
    return alpha, metric[0].__name__, np.array(best_score).mean(), best_model
