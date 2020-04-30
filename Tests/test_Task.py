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
from Manteia.Task import Task
from Manteia.Preprocess import Preprocess

class ManteiaTest(unittest.TestCase):

	"""Test case used to test the functions of the 'Manteia' module."""

	def test_Task(self):
		print("Test the functioning of the class Manteia.Task.")
		task=Task()
        
		# Vérifie que 'mant' est une 'str'
		self.assertEqual(type(task.test()), str)

	def test_Preprocess(self):
		print("Test the functioning of the class Manteia.Preprocess.")
		pp=Preprocess()
        
		# Vérifie que 'mant' est une 'str'
		self.assertEqual(type(pp.test()), str)
        
unittest.main()
