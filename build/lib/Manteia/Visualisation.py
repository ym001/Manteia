#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  data.py
#  
#  Copyright 2017 yves <yves.mercadier@ac-montpellier.fr>
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
#  '
from .Preprocess import Preprocess
from .Statistic import Statistic
from .Model import Model

import numpy as np
import matplotlib.pyplot as plt

class Visualisation:
	
	
	def __init__(self,path='',name='',save=True,show=False):
		self.path=path
		self.name=name
		self.save=save
		self.show=show
	def test(self):
		return "Visualisation Mantéïa."
	#Format de donnée en entrée un dictionnaire en sortie deux liste x,y 
	def format_data(self,dictionary=None):
		self.x=[]
		self.y=[]
		for key, value in dictionary.items():
			self.x.append(key)
			self.y.append(value)
			
	def plot_bar(self):
		#Plot the data:
		palette = plt.cm.get_cmap('tab10')

		plt.title(self.name)
		#Set tick colors:
		ax = plt.gca()
		ax.tick_params(axis='x', colors='gray')
		ax.tick_params(axis='y', colors='blue')
		
		plt.barh(self.x, self.y,color=palette.colors)
		if self.save:
			plt.savefig(self.path)
		if self.show:
			plt.show()
		
