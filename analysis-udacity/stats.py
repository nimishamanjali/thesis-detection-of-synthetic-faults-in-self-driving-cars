# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm

from patsy import dmatrices
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2 + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result


# calculates whether two accuracy arrays are statistically different according to GLM
def is_diff_sts(orig_value_list, mutant_value_list, cc, threshold=0.05):
    p_value = p_value_glm(orig_value_list, mutant_value_list)
    effect_size = cohen_d(orig_value_list, mutant_value_list)
    #if cc < 0:
    is_sts = ((p_value < threshold) and effect_size <= -0.5)
    #elif cc > 0:
     #   is_sts = ((p_value < threshold) and effect_size >= 0.5)
    return is_sts, p_value, effect_size


def p_value_glm(orig_value_list, mutant_value_list):
    zeros_list = [0] * len(orig_value_list)
    ones_list = [1] * len(mutant_value_list)
    mod_lists = zeros_list + ones_list
    acc_lists = orig_value_list + mutant_value_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)

    try:
        glm_results = glm.fit()
    except PerfectSeparationError:
        p_value_glm = 0
    else:
        glm_sum = glm_results.summary()
        pv = str(glm_sum.tables[1][2][4])
        # p_value = float(pv)
        p_value_glm = float(pv)

    return p_value_glm
