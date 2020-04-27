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
from Manteia import Manteia 
			
def main(args):
	documents = ["The !??    :::13New york ain't important feet geese a hell of a town ;;    //   ",
			"That's one small step for a man, one giant leap for mankind",
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
			["new york"],["moon"],["new york"],["new york"],["new york"],["new york"],
			["new york"],["london"],["london"],["london"],
			["london"],["london"],["new york"],["london"],["new york"],
			["london"],["london"],["london"],["london"],["new york"],["new york"]
			]
			
	mant=Manteia(documents,labels)
	print(mant.data.list_labels)
	print(mant.data.get_df())
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
