# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:56:09 2022

@author: User
"""

import math
import sys
sys.path.insert(1, r'C:\Users\User\OneDrive - UGent\python_functions')


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))