#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  test_Manteia.py
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

import unittest
from Manteia.Preprocess import Preprocess

class PreprocessTest(unittest.TestCase):

	"""Test case used to test the functions of the 'Manteia' module."""

	def Preprocess_test(self):
		print("Test the functioning of the class Manteia.Preprocess.")
		pp=Preprocess()
        
		# Vérifie que 'mant' est une 'str'
		self.assertEqual(type(pp.test()), str)

	def Preprocess_init(self):
		print("Test the functioning of the class Manteia.Preprocess.")
		pp=Preprocess(documents=['a','b'],labels=['a','b'])
        
		self.assertEqual(len(pp.documents), 2)
        
unittest.main()
