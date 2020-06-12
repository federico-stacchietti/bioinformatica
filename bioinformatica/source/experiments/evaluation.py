from .utils import metrics


def evaluate(scores, statistical_test, metric, alpha):
    best_model, best_score = [], 0
    for i, first_model in enumerate(scores):
        best_model, best_score = first_model, getattr(first_model, metric[0].__name__)
        for j, second_model in enumerate(scores):
            comparison_score = getattr(second_model, metric[0].__name__)
            if not i == j:
                stats, p_value = statistical_test(best_score, comparison_score)
                if p_value < alpha:
                    if comparison_score.mean() < best_score.mean():
                        best_model, best_score = second_model, comparison_score
    return alpha, metric[0].__name__, best_score, best_model
