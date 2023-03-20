# PyMicrofluidics

PyMicrofluidics is a python module that allows you to create create complex DXF designs. It is specifically written
for mirofluidics designs as it contains several pre-made classical microfluidics features in the form of parametrical
functions (e.g. a serpentine of adjustable size, alignment markers). It also offers the possibility to easily
create channels with rounded contours and fixed width, a feature often necessary in microfluidics applications.
In addition, it allows one to easily handle the case of multi-layer design, where different layers are often printed
one the two halves of a mask.

Two modules are available: mdfdesign and mfplotting. mfdesign contains the two main classes Design and Features handling
general features of the project and specific drawing implementations respectively. mfplotting allows to directly
visualize the design in a Jupyter notebook. It requires the installation (not done automatically) of the bokeh package.

## Installation

You can install this package directly from GitHub using the following command:

```
pip install "git+https://github.com/guiwitz/PyMicrofluidics.git"
```

To install the last version before the packaging update, use:
```
pip install "git+https://github.com/guiwitz/PyMicrofluidics.git@v0.2.4#egg=pymicrofluidics&subdirectory=pymicrofluidics"
```

Alternatively you can clone or download this repository and install the package locally. For that, move to the ```PyMicrofluidics``` folder and use:

```pip install .```

or

```pip install . --upgrade```

to install an updated version of the package.

The modules are then accessible using e.g.:
```python
 from pymicrofluidics.mfdesign import Design
```

### GDS format
If you want to save your design in GDS format, you need to install an additional package. You can do this by using:

```
pip install "git+https://github.com/guiwitz/PyMicrofluidics.git[gds]"
```

## Required packages
- numpy  
- shapely  
- dxfwrite
- bokeh

## Example

```python
import numpy as np

from pymicrofluidics.mfdesign import Design
from pymicrofluidics.mfdesign import Feature

design = Design()
design.add_layer('Layer1', {'name':'first_layer','color':1, 'inversion':0})
design.add_layer('Layer2', {'name':'second_layer','color':2, 'inversion':0})

polygon = Feature.define_polygon([[2*np.sin(2*np.pi/20*x)-5,2*np.cos(2*np.pi/20*x)+5] for x in range(20)])

serpentine = Feature.serpentine(nbseg = 4,dist = 5, rad = 1, length = 20, curvature = 2, origin = [0,0], orientation = 'horizontal', left_right = 'left', bottom_top = 'bottom')

text = Feature.define_text([0,-3],'This is my text')

polygon.set_layer('Layer1')
serpentine.set_layer('Layer2')
text.set_layer('Layer1')

design.add_feature('mypolygon', polygon)
design.add_feature('myserpentine', serpentine)
design.add_feature('mytext', text)

#choose a file location and name 
design.file = './example.dxf'

#draw full design
design.draw_design()

#save in GDS format. Only works if gdstk is installed (see above)
design.draw_gds('./example_gds.gds')

#close the drawing
design.close()
```
