#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Manteia

from setuptools import setup, find_packages
 
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
 
    name='Manteia',
 
    #version='0.0.10',
    version=Manteia.__version__,
 
    packages=find_packages(),
 
    author="Yves Mercadier",
 
    author_email="manteia.ym001@gmail.com",
 
    description="deep learning,NLP,classification,text,bert,distilbert,albert,xlnet,roberta,gpt2",
 
    long_description=open('README.md').read(),

    long_description_content_type='text/markdown',

    install_requires=requirements ,
 
    # Active la prise en compte du fichier MANIFEST.in
    include_package_data=True,
 
    url='https://github.com/ym001/Manteia',
 
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Communications",
    ],
 
 
    # Par exemple, si on veut créer la fabuleuse commande "proclame-sm", on
    # va faire pointer ce nom vers la fonction proclamer(). La commande sera
    # créé automatiquement. 
    # La syntaxe est "nom-de-commande-a-creer = package.module:fonction".
    entry_points = {
        'console_scripts': [
            'Manteia-test = Manteia.Manteia:testManteia',
            'Manteia-data = Manteia.Manteia:readData',
            'Manteia-classification = Manteia.Manteia:makeClassification',
        ],
    },
 

    #license="WTFPL",
 
)
