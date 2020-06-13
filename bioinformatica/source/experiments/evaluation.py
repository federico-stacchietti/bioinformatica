def test_models(models, statistical_test, metric, alpha):
    best_model = []
    for i, current_model in enumerate(models):
        best_model, current_model = current_model, current_model.scores.get(metric[0].__name__)
        for j, comparison_model in enumerate(models):
            comparison_scores = comparison_model.scores.get(metric[0].__name__)
            if not i == j:
                stats, p_value = statistical_test(current_model, comparison_scores)
                if p_value < alpha:
                    if comparison_scores.mean() < current_model.mean():
                        best_model = comparison_model
    return alpha, metric[0].__name__, best_model
