
Simple Documentation Tutorial : Mantéïa
=======================================

Designing your neural network to natural language processing. Deep learning has been used extensively in natural language processing (NLP) because it is well suited for learning the complex underlying structure of a sentence and semantic proximity of various words. Data cleaning
, construction model (Bert, Roberta, Distilbert, XLNet, Albert, GPT, GPT2), quality measurement training and finally visualization
of your results on several dataset ( 20newsgroups, Agnews Amazon Review Full, Amazon Review Polarity, DBPedia, PubMed_20k_RCT, Short_Jokes, Sogou News, SST-2, SST-5, Tweeter Airline Sentiment, Yahoo! Answers, Yelp Review Full, Yelp Review Polarity).

.. _Train your neural network: https://github.com/ym001/Manteia/blob/master/notebook/notebook_Manteia_classification_visualisation.ipynb
.. _Explore your data: https://github.com/ym001/Manteia/blob/master/notebook/notebook_Manteia_classification1.ipynb

+---------------------------------+------------------------------+
| `Train your neural network`_.   | `Explore your data`_.        | 
+=================================+==============================+
| .. image:: images/train.png     | .. image:: images/boxplot.png| 
+---------------------------------+------------------------------+

Installation
------------

You can install it with pip :

     pip install Manteia

.. _Anaconda: https://www.anaconda.com/open-source>

For use with GPU and cuda we recommend the use of `Anaconda`_. :

     conda create -n manteia_env python=3.7

     conda activate manteia_env

     conda install pytorch

     pip install manteia



Classes
=======

.. toctree::
   :maxdepth: 2

   ActiveLearning
   Augmentation
   Classification
   Dataset
   Generation
   Model
   Preprocess
   Statistic
   Task
   Visualisation


Link
====

.. toctree::
   :maxdepth: 2

   Notebooks<https://github.com/ym001/Manteia/tree/master/notebook>
   Exemples<https://github.com/ym001/Manteia/tree/master/Exemples>
   help
   Pypi<https://pypi.org/project/Manteia/>
   Source<https://github.com/ym001/Manteia>
   Documentation<https://manteia.readthedocs.io/en/latest/#>
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


