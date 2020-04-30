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
from Manteia.Statistic import Statistic
from Manteia.Model import Model
from Manteia.Classification import Classification
from Manteia.Visualisation import Visualisation

class ManteiaTest(unittest.TestCase):

	"""Test case used to test the functions of the 'Manteia' module."""

	def test_Task(self):
		print("Test the functioning of the class Manteia.Task.")
		task=Task()
        
		self.assertEqual(type(task.test()), str)

	def test_Preprocess(self):
		print("Test the functioning of the class Manteia.Preprocess.")
		pp=Preprocess()
        
		self.assertEqual(type(pp.test()), str)

	def test_Statistic(self):
		print("Test the functioning of the class Manteia.Statistic.")
		stat=Statistic()
        
		self.assertEqual(type(stat.test()), str)

	def test_Model(self):
		print("Test the functioning of the class Manteia.Model.")
		model=Model()
        
		self.assertEqual(type(model.test()), str)

	def test_Classification(self):
		print("Test the functioning of the class Manteia.Classification.")
		cl=Classification()
        
		self.assertEqual(type(cl.test()), str)

	def test_Visualisation(self):
		print("Test the functioning of the class Manteia.Visualisation.")
		visu=Visualisation()
        
		self.assertEqual(type(visu.test()), str)
        
unittest.main()
