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
import pandas as pd 
from Data import Data

			
def main(args):
	documents = ["!??    :::new york is important geese a hell of a town ;; //",
			"london is in the uk",
			"new york was originally dutch",
			"the big apple is great",
			"new york is also called the big apple",
			"nyc is nice",
			"people abbreviate new york city as nyc",
			"the capital of great britain is london",
			"london is in england",
			"london is in great britain",
			"it rains a lot in london",
			"london hosts the british museum",
			"new york is great and so is london",
			"i like london better than new york",
			'nice day in nyc',
			'welcome to london',
			'london is rainy',
			'it is raining in britian',
			'it is raining in britian and the big apple',
			'it is raining in britian and nyc',
			'hello welcome to new york. enjoy it here and london too']
			
	labels = [
			["new york"],["london"],["new york"],["new york"],["new york"],["new york"],
			["new york"],["london"],["london"],["london"],
			["london"],["london"],["new york"],["london"],["new york"],
			["london"],["london"],["london"],["london"],["new york"],["new york"]
			]
			
	data=Data(documents,labels)
	print(data.list_labels)
	print(data.get_df().head())
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
