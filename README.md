# Assignment 1. Segmentation



## Installation

Install the python package in editable mode with
```bash
pip install -e .
```

The code was made and adopted in [Pycharm](https://www.jetbrains.com/pycharm/) IDE.

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
    --opt True            # a flag to indicate whether we use optimized version of meanshift algorithm or not
    
```
### 2. Running the actual image segmentation script
An example of running the image segmentation script. 
```
# Note: provide an image in .jpg format having 3 RGB channels.

python Code/imageSegmentation.py
    --im Data/img3.jpg    # path to the image
    --r 2                 # the search window radius
    --c 4                 # some  constant  value (c = 4 by default)
    --feature_type 5      # the dimension of the feature vector as 3D or as 5D (specifying the color and x, y coordinates for each pixel)
    --cloud True          # flag to indicate whether we plot a 3D cloud of points distribution
    --opt False           # a flag to indicate whether we use optimized version of meanshift algorithm or not
    
```

### 3. Running the filtering script
An example of running the filtering script. 
```
# Note: provide an image in .jpg format having 3 RGB channels.

python Code/filtering.py
    --im Data/img3.jpg    # path to the image
    
```

### 4. Running the scripts on the Google Colab platform
- Create a new notebook on Google Colab. 
- Clone the repository executing in a cell:
```
git clone https://github.com/CyrilShch/Kirill_Shcherbakov_First_Assignment_CV2021.git
```
- Move to the directory of the cloned repository executing in a cell:
```
cd Kirill_Shcherbakov_First_Assignment_CV2021
```
- Run a script. For example, execute in a cell:
```
# Note: provide an image in .jpg format having 3 RGB channels.

python Code/imageSegmentation.py
    --im Data/img3.jpg    # path to the image
    --r 2                 # the search window radius
    --c 4                 # some  constant  value (c = 4 by default)
    --feature_type 5      # the dimension of the feature vector as 3D or as 5D (specifying the color and x, y coordinates for each pixel)
    --cloud True          # flag to indicate whether we plot a 3D cloud of points distribution
    --opt False           # a flag to indicate whether we use optimized version of meanshift algorithm or not
    
```
