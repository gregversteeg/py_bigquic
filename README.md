# Run BigQUIC Graphical LASSO (GLASSO) solver in Python

1. Install [R](https://www.r-project.org).
2. Run this command within R.
```R
install.packages('BigQuic', repos='http://cran.us.r-project.org')
```
3. Install rpy2 to call R from python (from the command line). (Also pip install sklearn and numpy if you don't already have them.) 
```
pip install rpy2
```
4. Run `python py_bigquic.py` to test. 
5. An example usage is below. 
```python
import numpy as np
import py_bigquic as pbq

data_array = np.random.random((10,20))  # A test data array with 10 samples, 20 variables
alpha = 0.5  # The sparsity hyper parameter
prec = pbq.bigquic(data_array, alpha)  # Returns the precision matrix
```

I implemented this to do large-scale comparisons on high-dimensional, under-sampled covariance estimation
with [Linear CorEx](https://github.com/gregversteeg/LinearCorex). 
If you are interested in this type of application, I strongly recommend checking Linear CorEx out, as
it outperforms GLASSO in many scenarios! Results will be added to this paper:
[Low Complexity Gaussian Latent Factor Models and a Blessing of Dimensionality](https://arxiv.org/abs/1706.03353).