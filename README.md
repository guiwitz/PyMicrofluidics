# PyMicrofluidics
PyMicrofluidics is a python module that allows you to create create complex DXF designs. It is specifically written for mirofluidics designs as it contains several pre-made classical microfluidics features in the form of parametrical functions (e.g. a sperpentine of adjustable size, alignement markers). It also offers the possibility to easily create channels with rounded contours and fixed width, a feature often necessary in microfluidics applications. In addition, it allows one to easily handle the case of multi-layer design, where different layers are often printed one the two halves of a mask.

## Required packages
scipy
numpy
shapely
dxfwrite

##Example

```python
from dxfwrite import DXFEngine as dxf
import numpy as np

from microfluidics_class import Design
from microfluidics_class import Feature

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

#close the drawing
design.close()
```
