# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 08:30:17 2016

@author: konopczynski
Visualize the output volume
"""

import numpy as np
import pyqtgraph as pg
import sys
sys.path.append('../')
import config


def main():
    param = config.read_parameters()
    path2save = param.path2Output
    fn = param.dictionaryName+'_'+param.outputFn
    z = np.load(path2save+fn)
    pg.image(z)
    raw_input('click to end')
    return None

if __name__ == '__main__':
    main()
