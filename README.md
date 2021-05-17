# Assignment-1--Segmentation



## Installation

Install the python package in editable mode with
```bash
pip install -e .
```

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scipy](https://www.scipy.org/)
- [Scipy.io](https://docs.scipy.org/doc/scipy/reference/io.html)
- [Collections](https://docs.python.org/3/library/collections.html)
- [Skimage](https://scikit-image.org/)

### 1. Debugging the implementations on the toy data (pts.mat)
An example of running the debugging script. 
```
python Code/debug_img.py
    --data Data/pts.mat   # the data path
    --r 2                 # the search window radius
    --c 4                 # some  constant  value (c = 4 by default)
    --opt False           # a flag to indicate whether we use optimized version of meanshift algorithm or not
