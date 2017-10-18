from bokeh.plotting import figure, output_notebook, show
from ipywidgets import SelectMultiple
from ipywidgets import interactive
from ipywidgets import interact, fixed
from ipywidgets import Checkbox
from bokeh.models import Range1d
from bokeh.plotting import figure, output_notebook, show

import numpy as np

def layer_select(m, design):

    layers_to_plot = m
    features_to_plot = {'features':[],'layer':[]}
    if len(layers_to_plot)>0:
        features_to_plot['features'] = [x for x in design.features if design.features[x].layer in layers_to_plot]
        features_to_plot['layers'] = [layers_to_plot.index(design.features[x].layer) for x in design.features if design.features[x].layer in layers_to_plot]

        if len(features_to_plot['features'])>0:
            #chk = [Checkbox(description=a) for a in features_to_plot]
            #interact(updatePlot, **{c.description: c.value for c in chk})
            sel_mult2 = SelectMultiple(options = features_to_plot['features'])
            interactive_plot2 = interactive(updatePlot, n=sel_mult2, 
                                            design = fixed(design), all_features = fixed(features_to_plot))
            display(interactive_plot2)
            
def find_min(cur_min, new_min):
    if new_min<cur_min:
        cur_min = new_min
    return cur_min

def find_max(cur_max, new_max):
    if new_max>cur_max:
        cur_max = new_max
    return cur_max

def update_bounds(cur_bound, feature_coord):
    cur_bound[0] = find_min(cur_bound[0],np.min([np.min(x[:,0]) for x in feature_coord]))
    cur_bound[1] = find_max(cur_bound[1],np.max([np.max(x[:,0]) for x in feature_coord]))

    cur_bound[2] = find_min(cur_bound[2],np.min([np.min(x[:,1]) for x in feature_coord]))
    cur_bound[3] = find_max(cur_bound[3],np.max([np.max(x[:,1]) for x in feature_coord]))

    return cur_bound


def updatePlot(n, all_features, design):
    colors = ['red','blue']
    #clear_output()
    features = n
    if len(features)>0:
        p = figure(title="simple line example", x_axis_label='x', y_axis_label='y',output_backend="webgl")
        cur_bound = [100000,-100000,100000,-100000]
        
        for elem in features:
            layer_index = all_features['layers'][all_features['features'].index(elem)]
            
            toplot = design.features[elem].coord
            cur_bound = update_bounds(cur_bound, toplot)
            for tp in toplot:
                p.line(tp[:,0], tp[:,1],line_color=colors[layer_index])
            #if not design.features[elem].mirror == None:
            #    flipped = design.features[elem].flip_feature(0).coord
            #    cur_bound = update_bounds(cur_bound, flipped)
            #    for tp in flipped:
            #        p.line(tp[:,0], tp[:,1])
                    
        maxrange = np.max([cur_bound[1]-cur_bound[0],cur_bound[3]-cur_bound[2]])
        p.x_range=Range1d(cur_bound[0],cur_bound[0]+maxrange)
        p.y_range=Range1d(cur_bound[2],cur_bound[2]+maxrange)
                
        plot_height = 800
        plot_width = 800
        
        #plot_width = int(800*((cur_bound[1]-cur_bound[0])/(cur_bound[3]-cur_bound[2])))
        #plot_height = 800
        #if plot_width>800:
        #    p.plot_height = int(plot_height/(plot_width/800))
        #    p.plot_width = 800
        #elif plot_height>800:
        #    p.plot_width = int(plot_height/(plot_height/800))
        #    p.plot_height = 800
        #else:
        #    p.plot_width = plot_width
        #    p.plot_height = plot_height
        
        #p.plot_height = 100
        #p.inner_height = 100
        #p.inner_width  = 100
        
        show(p)
        
def plot_design(design):
    output_notebook()
    sel_mult = SelectMultiple(options = [x for x in design.layers])
    interactive_plot = interactive(layer_select, m=sel_mult, design=fixed(design))
    display(interactive_plot)