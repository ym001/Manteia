#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  exemple_Data.py
#  
#  Copyright 2020 Yves <yves@mercadier>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#
from Manteia.Classification import Classification 
from Manteia.Preprocess import Preprocess
from Manteia.Dataset import Dataset
			
def main(args):
	
	ds        = Dataset('20newsgroups')
	documents = ds.get_documents()
	labels    = ds.get_labels()
	pp        = Preprocess(documents=documents,labels=labels,nb_sample=50)
	cl        = Classification(documents=pp.documents,labels=pp.labels,process=True)
	print(cl.predict(documents[:5]))

	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
