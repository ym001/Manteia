#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  core.py
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
 
"""
	This module proclaims the good word. May they
	regain total freedom of artificial thought towards a new age
	reminiscent.

	You can install it with pip:

		pip install Manteia

	Example of use:

	>>> from Manteia import testManteia
	>>> testManteia ()

	This code is licensed under MIT.
"""
__all__ = ['testManteia','testData','testClassification']

from .Preprocess import Preprocess
from .Classification import Classification
from .Statistic import Statistic
from .Visualisation import Visualisation
from .Model import Model

class Task:
	def __init__(self,documents=None,labels=None,task='classification'):
		if documents!=None:
			self.data=Data(documents,labels)
		if task=='classification' and documents!=None:
			self.classification=Classification(data=self.data)
	def test(self):
		return "Hello, Task Mantéïa is alive."
		
def testManteia():
    print ("Hello, Mantéïa is alive.")

def testData():
	documents=['    ,;:123test   car','test houses']
	labels=['1','0']
	mant=Data(documents,labels)
	print(mant.data.list_labels)
	print(mant.data.get_df())

def testClassification():
	documents=['test car','test house']
	labels=['1','0']
	mant=Classification(documents,labels)

