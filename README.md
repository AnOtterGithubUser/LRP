*This project is a work in progress, many features may be added/removed/
changed in the coming month*

# PyExplain

Why does my neural net predict this ?  
How many times have you heard that neural networks are "black-boxes" and
as such their usage is severely limited in safety critical applications ? 
It would be a shame not to use these powerful models for such reason.  
PyExplain is a project that aims at making neural nets decision interpretable with 
a single line of code. It is based on Deep Taylor Decomposition, a method 
that back-propagates the relevance of each neurons for the decision.  
Our implementation is in PyTorch.

# Authors
This project is an initiative of Thomas PESNEAU and Amine SABONI

# Bibliography
[1] Explaining nonlinear classification decisions with Deep Taylor Decomposition, G. Montavon, S. Bach, A. Binder, W. Samek, K-R Muller, 2015 