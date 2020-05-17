#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Manteia.Statistic import Statistic
from Manteia.Dataset import Dataset

def main(args):
	
	ds=Dataset('pubmed_rct20k')

	Statistic(ds.documents_train,ds.labels_train)
	
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
