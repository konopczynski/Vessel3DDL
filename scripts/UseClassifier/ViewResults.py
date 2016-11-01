# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 08:30:17 2016

@author: konopczynski
Visualize the output volume
"""

import sys
import os
import numpy as np
import pyqtgraph as pg
import config as C

if __name__ == '__main__':
    Param = C.ReadParameters()
    Path2Save =  Param.path2Output
    fn = Param.outputFn 
    z = np.load(Path2Save+fn)
    pg.image(z)
    raw_input('click to end')
