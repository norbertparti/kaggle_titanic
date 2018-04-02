#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 18:09:34 2018

@author: nparti
"""

import numpy as np
import scipy.stats  as stats

samples = np.array([2,2,2,3,3,3])

unique, counts = np.unique(samples, return_counts=True)

propbability_of_2 =  counts[0] / len(samples)

propbability_of_3 =  counts[1] / len(samples)

entropy = stats.entropy([2/3, 1/3], base=2.0)

print('Entropy of samples {}'.format(entropy))