from pymicrofluidics.mfdesign import Design

def test_something():
    pass

def test_design_add_layer():
    design = Design()

    design.add_layer('Layer1', {'name':'first_layer','color':1, 'inversion':0})
    design.add_layer('Layer2', {'name':'sec_layer','color':2, 'inversion':0})

    assert len(design.layers) == 2
