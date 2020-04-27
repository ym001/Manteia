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
    Implémentation de la proclamation de la bonne parole.
 
    Usage:
 
    >>> from Manteia import testManteia
    >>> testManteia()
"""
__all__ = ['testManteia']

from Data import Data
from Classification import Classification

class Manteia:
	def __init__(self,documents=None,labels=None,task='classification'):
		self.data=Data(documents,labels)
		if task=='classification':
			self.classification=Classification(data=self.data)

def testManteia():
    print ("Hello, Mantéïa is alive.")

def readData():
	documents=['test voiture','test maison']
	labels=['1','0']
	mant=Manteia(documents,labels)
	print(mant.data.list_labels)
	print(mant.data.get_df())

def makeClassification():
	documents=['test voiture','test maison']
	labels=['1','0']
	mant=Manteia(documents,labels)
	
