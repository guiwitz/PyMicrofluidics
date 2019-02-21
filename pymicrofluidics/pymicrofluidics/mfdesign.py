import numpy as np
#import matplotlib.pyplot as plt
from dxfwrite import DXFEngine as dxf

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely import ops

import copy
import re
import warnings

import pkg_resources

#import scipy.interpolate
#from scipy.interpolate import CubicSpline


def get_hershey():
    """
        Parsing of a hershey table to generate numbers. This is only for internal purposes.
    """
    hershey_path = pkg_resources.resource_filename('pymicrofluidics', 'data/hershey.txt')
    hershey_table = {}
    first = True
    with open(hershey_path) as openfileobject:
        for tline in openfileobject:
            if re.search('Ascii',tline):
                if first == False:
                    newline = hershey_table[asci]['coord'].split('-1,-1,')
                    newline = [list(filter(None, x.split(','))) for x in newline if len(x)>0]
                    hershey_table[asci]['coord'] = [np.array([[float(y[x]),float(y[x+1])] for x in range(0,len(y)-1,2)])/21 for y in newline]
                    if len(hershey_table[asci]['coord'])>0:
                        middle = 0.5*(np.max(np.concatenate(hershey_table[asci]['coord'])[:,0])+np.min(np.concatenate(hershey_table[asci]['coord'])[:,0]))
                        #middle = float(middle)
                        hershey_table[asci]['coord'] = [np.array([[x[0]-middle,x[1]] for x in y]) 
                                                        for y in hershey_table[asci]['coord']]
                        hershey_table[asci]['width'] = np.max(np.concatenate(hershey_table[asci]['coord'])[:,0])-np.min(np.concatenate(hershey_table[asci]['coord'])[:,0])
                    else:
                        hershey_table[asci]['width'] = 0.5
                asci = int(re.findall('.*Ascii (\d+).*',tline)[0])
                width = float(re.findall('\d+,\s*(\d+),.*',tline)[0])
                hershey_table[asci] = {'coord': '', 'width': width}
                first = False
            else:
                newline = tline.rstrip('\n')
                hershey_table[asci]['coord'] = hershey_table[asci]['coord']+newline
    return hershey_table

class Design:
    
    def __init__(self):
        
        self.features = {}
        self.layers = {}
        self.file = None
        #Feature.design = self
        
    
    def __add__(self, other):
        sum_design = copy.deepcopy(self)
        for l in other.layers:
            if l in sum_design.layers.keys():
                if other.layers[l] != sum_design.layers[l]:
                    raise Exception("Trying to add two designs where layers with the same name aren't identical.") 
            sum_design.layers[l] = other.layers[l]
        for f in other.features:
            new_name = f
            suffix = 0
            while new_name in sum_design.features.keys():
                new_name = new_name+str(suffix)
                suffix = suffix+1
            sum_design.features[new_name] = other.features[f]
        return sum_design
                
        
        
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    def copy(self):
        """
        Returns a copy of a design.
        
        This makes a deep copy to ensure designs are trully independent.

        Parameters
        ----------
        
        Returns
        -------
        design
            copied design

        """
        return copy.deepcopy(self)
        
        
    def add_layer(self, layer_name, layer):
        self.layers[layer_name] = layer
        
    def add_feature(self, name, feature):
        self.features[name] = feature
        
        if not feature.mirror == None:
            self.features[name+'_mirror'] = feature.mirror_feature(y=feature.mirror)
                
    def get_coord(self,name):
        return self.features[name].coord
    
    def add_to_coord(self,name,item,toadd):
        self.features[name].coord[item] = np.array([x+np.array(toadd) for x in self.features[name].coord[item]])
        
    def multiplicate(self, positions):
        #complete_design = Design()
        #complete_design.layers = self.layers
        #complete_design.file = self.file
        
        complete_design = sum([self.copy().set_design_origin(p) for p in positions])
        
        #for p in positions:
        #    original = copy.deepcopy(self)
        #    original.set_design_origin(p)
        #    complete_design = complete_design + self.copy().set_design_origin(p)
            #for f in original.features:
            #    original.features[f].mirror = None
            #    complete_design.add_feature(f+str(counter),original.features[f])
            #counter  = counter +1
        return complete_design
                
                
        
    def set_design_origin(self,origin):
        """
        Moves all the features of a desing to a chosen origin

        All features are list of 2D numby array stored as dictionary enries of desing.
        This function adds to each coordinate of every feature the vector origin

        Parameters
        ----------
        design : dictionary 
            dictionary whose entries are features
        origin : 2D numpy array
            Point to which the design should be moved to

        Returns
        -------
        dxfwrite object
            Same drawing as input but with new features moved

        """
        for d in self.features:
            for i in range(len(self.get_coord(d))):
                self.add_to_coord(d,i,origin)
        return self
    

    def add_polyline(self, layer_to_use,poly,open, drawing):
        """
        Adds a feature to the DXF drawing.

        Adds a feature (numpy polygon or list of numpy polygons) in a chosen layer 
        of a DXF drawing opened using the dxfwrite package.

        Parameters
        ----------
        layer_to_use : dictionary 
            dictionary with field 'name' of the layer 
        poly : list of 2D positions lists
            list of polygons coordinates
        open : boolean
            True means the feature is text (open polygon)
        drawing : dxfwrite object
            dxfwrite object openend unsing drawing()  method

        Returns
        -------
        dxfwrite object
            Same drawing as input but with new feature added

        """
        if type(poly) is not list:
            toplot = [poly]
        else:
            toplot = poly

        for y in toplot:
            polyline= dxf.polyline(layer=layer_to_use['name'])
            if open==True:
                polyline.close(False)
            else:
                polyline.close(True)
            y = np.round(100*y)/100
            if layer_to_use['inversion']==0:
                polyline.add_vertices(y)
            else:
                polyline.add_vertices(-y)
            drawing.add(polyline)

        return drawing
    
        
    def draw_design(self):
        """
        Adds all features of the deisgn into the drawing

        All features stored as dictionary enries of desing are written in the dxf file 
        drawing opened with the drawing() method of the dxfwrite package

        Parameters
        ----------
        design : dictionary 
            dictionary whose entries are features
        drawing : dxfwrite object
            dxfwrite object openend unsing drawing()  method

        Returns
        -------
        dxfwrite object
            Same drawing as input but with new feature added

        """
        drawing = dxf.drawing(self.file)
        
        for x in self.layers:
            drawing.add_layer(self.layers[x]['name'], color=self.layers[x]['color'])
        
        for x in self.features:
            #if self.features[x].open == False:
            self.add_polyline(self.layers[self.features[x].layer],self.features[x].coord,
                                     self.features[x].open,drawing)
            #else:
                #self.add_text(self.layers[self.features[x].layer],self.features[x].coord,self.features[x].text,drawing)
                #self.add_polyline(self.layers[self.features[x].layer],self.features[x].coord,drawing)
            
        self.drawing = drawing
        #return drawing
        
    def close(self):
        self.drawing.save()
        
    
    
class Feature:
    
    hershey_table = get_hershey()
    
    def __init__(self, coord = None, layer=None, mirror=None, open = False, text=None):
        
        self.coord = coord
        self.layer = layer
        self.mirror = mirror
        self.open = open
        self.text = text
        
    def __add__(self, other):
        sum_feature = copy.deepcopy(self)
        sum_feature.coord = sum_feature.coord+other.coord
        return sum_feature
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    

    def set_layer(self,layer_name):
        
        #self.layer = self.design.layers[layer_name]
        self.layer = layer_name
        return self
    
    def set_mirror(self,ax):
        
        self.mirror = ax
        return self
    
    @classmethod    
    def define_polygon(cls, polygon):
        """
        Generates a polygon feature

        Parameters
        ----------
        polygon : 2D list 
            polygon to be defined as feature

        Returns
        -------
        feature
            polygon feature

        """
        
        num_obj = cls()
        num_obj.coord = [np.array(polygon)]
        return num_obj
    
    
    @classmethod 
    def define_text(cls, position, text, scale=1, rotation=0):
        """
        Generates a text feature

        Parameters
        ----------
        position : 2D point 
            where to write text
        text : string
            what to write
        scale : float
            scale of text (default height is 0 for capitals)
        rotation : float
            text rotation in radians

        Returns
        -------
        feature
            regular feature

        """
        
        position = np.array(position)
        text_obj = cls()
        text_obj.coord = []
        sep = 0.2*scale
        posref=np.array([0.0,0.0])
        
        #complete text length
        #textlen = 0.5*sep*(len(text)-1)+0.5*scale * np.sum([np.array(Feature.hershey_table[ord(text[x])]['width']) for x in range(len(text))])
        for x in range(len(text)):
            code = ord(text[x])
            letter = Feature.hershey_table[code]
            if x>0:
                posref[0]=posref[0]+ scale*np.array(letter['width'])/2+sep
            #posref[0] = posref[0]-textlen
            #position[0] = position[0] + scale*np.array(letter['width'])/2+sep
            for k in letter['coord']:
                if len(k)>0:
                    k = np.array(k)
                    coord = [scale*m+posref-np.array([0,0.5*scale]) for m in k]
                    text_obj.coord.append(np.squeeze(coord))
            posref[0]=posref[0]+ scale*np.array(letter['width'])/2+sep
            
        #align number
        posref[0]=posref[0]- scale*np.array(letter['width'])/2-sep
        for x in range(len(text_obj.coord)):
            text_obj.coord[x] = np.array([z- np.array([posref[0]/2,0]) for z in text_obj.coord[x]])
        
        #Rotate the number by alpha
        if rotation !=0:

            alpha = rotation
            R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])

            for i in range(len(text_obj.coord)):
                text_obj.coord[i] = np.squeeze([np.dot([x],R) 
                                                       for x in text_obj.coord[i]])
            
        #move the number to position pos
        for x in range(len(text_obj.coord)):
            text_obj.coord[x] = np.array([z+ position for z in text_obj.coord[x]])
        
        #make thick numbers (currently unused)
        '''for n in range(len(text_obj.coord)):
            x = text_obj.coord[n][:,0]
            y = text_obj.coord[n][:,1]
            x2 = np.array([[x[i],(2/3*x[i]+1/3*x[i+1]),(1/3*x[i]+2/3*x[i+1])] for i in range(x.shape[0]-1)])
            y2 = np.array([[y[i],(2/3*y[i]+1/3*y[i+1]),(1/3*y[i]+2/3*y[i+1])] for i in range(y.shape[0]-1)])
            x2 = np.concatenate(x2)
            x2 = np.append(x2,x[-1])
            y2 = np.concatenate(y2)
            y2 = np.append(y2,y[-1])
            x=x2
            y=y2

            #nt = np.linspace(0, 1, 40)
            #t = np.zeros(x.shape)
            #t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
            #t = np.cumsum(t)
            #t /= t[-1]
            #x2 = scipy.interpolate.spline(t, x, nt,order=2)
            #y2 = scipy.interpolate.spline(t, y, nt,order=2)
            newcoord = np.stack((x2,y2),axis=1)
            
            csx = CubicSpline(np.linspace(0, 1, x.shape[0]), x,bc_type = 'clamped')
            csy = CubicSpline(np.linspace(0, 1, y.shape[0]), y,bc_type = 'clamped')
            s
            xs = np.linspace(0, 1,1000)
            newcoord = np.stack((csx(xs),csy(xs)),axis=1)
  
            text_obj.coord[n]=Feature.define_tube(newcoord,0,100).coord[0]'''
        #set object as text type
        text_obj.open = True
        return text_obj
    
        
    @classmethod    
    def define_tube(cls, points,curvature, rad):
        """
        Generates polygon coordinates of a tube

        The tube goes along points, has a radius rad and each "turn" has a curvature.

        Parameters
        ----------
        points : 2D list 
            Tube path
        curvature : 2D list
            Tube curvature at each coordinate. First and last point must be 0
        rad : float
            Tube radius

        Returns
        -------
        2D numpy array
            Coordinates of polygon representing the tube

        """
        
        points = np.array(points)
        
        if (type(curvature) is int) or (type(curvature) is float):
            curvature = curvature*np.ones(points.shape[0])
            curvature[0]=0
            curvature[-1]=0
        curvature = np.array(curvature)
        
        #remove points whose distance is close to zero
        cond = np.concatenate((np.array([100]),np.linalg.norm(np.diff(points,axis=0),axis=1)))>0.000001
        #print(cond)
        points = points[cond]
        curvature = curvature[cond]
        
        #verify that curvatures are zero at both ends and that curvature is not larger than
        #the tube radius
        if curvature[0] !=0:
            curvature[0] = 0
            warnings.warn('Initial curvature was not 0')
        if curvature[-1] !=0:
            curvature[-1] = 0
            warnings.warn('Last curvature was not 0')
        
        
        #if the tube is made of more than 2 points, check curvatures/radius
        if points.shape[0]>2:
            adjusted_curvature = False
            if np.any(curvature[curvature>0]<rad/0.9):
                warnings.warn('Tube radius cannot be larger than curvature. Forcing larger curvatures')
                curvature[(curvature<rad/0.9)&(curvature>0)]=rad/0.9
                adjusted_curvature = True

            #verify that the chosen curvature are not too large to be accommodated on the segments
            distances, vecnorms = Feature.check_curvatures(points,rad,curvature)
            all_fact = np.empty((0,2))
            for i in range(points.shape[0]-1):
                #combined space occupied on a given segment by its two neighboring curved regions
                sum_dist1 = distances[i]+distances[i+1]
                #ratio of vector length and combined occupied region
                if sum_dist1>0:
                    all_fact=np.append(all_fact,np.array([[i,(vecnorms[i]/sum_dist1)]]),axis=0)
                else:
                    all_fact=np.append(all_fact,np.array([[i,2]]),axis=0)
            #sort the ratios from smallest (way to much space occupied by cureved region) to largest 
            order = all_fact[all_fact[:, 1].argsort()][:,0]
            #if any curved regions take too much space, reduce them by the calculated factor and rerun the 
            #calculation of occupied region (as they affect two segments). I do that in a sorted way to ensure
            #that I take first care of the worst cases to avoid over-correcting.
            if np.any(all_fact[:,1]<1):
                warnings.warn('Some curvatures are too large to be accommodated on the given segment lenghts and will be reduced')
                for i in order:#range(points.shape[0]-1):
                    i=int(i)
                    sum_dist1 = distances[i]+distances[i+1]
                    if sum_dist1>vecnorms[i]:
                        factor = 0.9*(vecnorms[i]/sum_dist1)
                        curvature[i]=factor*curvature[i]
                        curvature[i+1]=factor*curvature[i+1]
                        distances, vecnorms = Feature.check_curvatures(points,rad,curvature)

            if np.any(curvature[curvature>0]<rad/0.9):
                warnings.warn('The chosen combination of path and radius has no solution. The radius is modified!!!')
                rad = 0.9*np.min(curvature[curvature>0])
            
        complete = np.array([points[0,:]])
        for i in range(1,len(curvature)):
            if curvature[i] != 0:
                vec1 = (points[i+1,:]-points[i,:])/np.linalg.norm(points[i+1,:]-points[i,:])
                vec2 = (points[i-1,:]-points[i,:])/np.linalg.norm(points[i-1,:]-points[i,:])
                
                if vec1[0]*vec2[1]-vec1[1]*vec2[0] == 0:
                    complete = np.append(complete,[points[i,:]],axis=0)
                    continue
                
                bis = (vec1+vec2)/np.linalg.norm(vec1+vec2)
                
                crossprod = np.dot(vec1,vec2)
                if crossprod<-1:
                    crossprod = -1
                gamma = np.arccos(crossprod)/2
                
                D2 = curvature[i]/np.sin(gamma)
                alpha2 = np.pi/2-gamma


                if np.cross(vec1,vec2)==0:
                    complete = np.append(complete,points[i,:],axis=0)
                    continue

                if np.cross(vec1,vec2)<0:
                    angles = np.arange(-alpha2,alpha2,0.1)
                else:
                    angles = np.arange(alpha2,-alpha2,-0.1)
                if angles.shape[0]==0:
                    angles = np.array([0])

                center = points[i,:]+D2*bis

                vect = -curvature[i]*bis
                arc = np.squeeze([center+np.dot(vect,np.array([[np.cos(x),-np.sin(x)],[np.sin(x),np.cos(x)]])) for x in angles])
                
                if len(arc.shape)==1:
                    arc = np.array([arc])
                complete = np.append(complete,arc,axis=0)
            else:
                complete = np.append(complete,[points[i,:]],axis=0)

        vect = np.diff(complete,axis=0)
        vect = np.array([vect[x,:]/np.linalg.norm(vect[x,:]) for x in range(vect.shape[0])])
        vect = np.concatenate((np.transpose([vect[:,1]]),np.transpose([-vect[:,0]])),axis=1)

        tube1 = complete[0:-1,:]+rad*vect
        tube1 = np.append(tube1,[complete[-1,:]+rad*vect[-1,:]],axis=0)
        tube2 = complete[0:-1,:]-rad*vect
        tube2 = np.append(tube2,[complete[-1,:]-rad*vect[-1,:]],axis=0)

        tube = np.append(tube1, np.flipud(tube2),axis=0)


        tube = np.round(tube*100)/100
        tube_obj = cls()
        tube_obj.coord = [tube]
        return tube_obj

    #this function is a helper for the tube definition. It checks whether curvatures are compatible with 
    #the chosnen path. In summary the extent of the curved region cannot be larger than the straight
    #segments themselves
    def check_curvatures(points,rad,curvature):

        #distances contains a list of the space occupied by the curved region of each vertex
        #vecnorms contains the length of each segment
        points = np.array(points)
        distances = np.array([0])
        vecnorms = np.array([])
        for i in range(1,len(curvature)-1):
            vecnorms = np.append(vecnorms,[np.linalg.norm(points[i-1,:]-points[i,:])],axis=0)
            if curvature[i] != 0:
                vec1 = (points[i+1,:]-points[i,:])/np.linalg.norm(points[i+1,:]-points[i,:])
                vec2 = (points[i-1,:]-points[i,:])/np.linalg.norm(points[i-1,:]-points[i,:])


                if vec1[0]*vec2[1]-vec1[1]*vec2[0] == 0:
                    distances = np.append(distances,[0],axis=0)
                    continue

                bis = (vec1+vec2)/np.linalg.norm(vec1+vec2)

                crossprod = np.dot(vec1,vec2)
                if crossprod<-1:
                    crossprod = -1
                gamma = np.arccos(crossprod)/2

                D2 = curvature[i]/np.sin(gamma)
                alpha2 = np.pi/2-gamma
                distances = np.append(distances,[np.sqrt(D2**2-curvature[i]**2)],axis=0)
            else:
                distances = np.append(distances,[0],axis=0)
        vecnorms = np.append(vecnorms,[np.linalg.norm(points[i+1,:]-points[i,:])],axis=0)
        distances = np.append(distances,[0],axis=0)
        return distances, vecnorms
    
    @classmethod    
    def define_tube_broken(cls, points,curvature, rad, dotlen):
        """
        Generates polygon coordinates of a tube

        The tube goes along points, has a radius rad and each "turn" has a curvature.

        Parameters
        ----------
        points : 2D list 
            Tube path
        curvature : float
            Tube curvature for each segment
        rad : float
            Tube radius
        dotlen : length of broken line segments

        Returns
        -------
        2D numpy array
            Coordinates of polygon representing the broken line tube

        """
        points = np.array(points)
        totlen = np.sum(np.linalg.norm(np.diff(np.array(points),axis=0), axis=1))
        interpol_points = np.array([np.array(LineString(tuple(points)).interpolate(x).coords).squeeze() for x in np.arange(0,totlen,dotlen/2)]).squeeze()
        
        broken_lines = [interpol_points[x:x+3] for x in range(0,interpol_points.shape[0]-4,4)]
        #print([x for x in broken_lines])
        broken_lines_feature = []
        #for x in broken_lines:
        #    broken_lines_feature.append(Feature.define_tube(x, [0,curvature,0], rad))
        #broken_lines_feature = sum(broken_lines_feature)
        
        broken_lines_feature = sum([Feature.define_tube(x, [0,curvature,0], rad) for x in broken_lines])
        return broken_lines_feature

    
    
    @classmethod   
    def serpentine(cls, nbseg, dist, rad, length, curvature, origin, orientation, left_right, bottom_top, prune_first=0, prune_last=0):
        """
        Generates a serpentine feature

        The serpentine has several properties. First, the number of segments, their length, radius, curvature
        and separation. Second the positioning and orientation of the serpentine.

        Parameters
        ----------
        nbseg : int 
            Number of segments composing the serpentine
        dist : float
            Separation between segments
        rad : float
            Tube radius
        curvature : float
            Curvature of turns
        origin: 2D list
            Position of the "start" of the serpentine
        orientation : serpentine is 'horizontal' or 'vertical'
        left_right : str
            serpentine starts 'left' or 'right'
        bottom_top : str
            serpentine starts 'top' or 'bottom'
        prune_first : int (optionnal)
            cut a given length of the first segment (append one if negative value)
        prune_last : int (optionnal)
            cut a given length of the last segment (append one if negative value)


        Returns
        -------
        mf_feature
            Coordinates of polygon representing the tube

        """

        origin = np.array(origin)
        serp = np.array([[0,0]])
        if left_right == 'left':
            serp_init1 = np.array([0,0])
            serp_init2 = np.array([length,0])
        else:
            serp_init1 = np.array([length,0])
            serp_init2 = np.array([0,0])

        toadd = np.array([0,dist])
        if bottom_top == 'top':
            toadd = -toadd

        for x in range(nbseg):
            if np.mod(x,2)==0:
                serp = np.append(serp, [serp_init1+x*toadd],axis=0)
                serp = np.append(serp, [serp_init2+x*toadd],axis=0)
            else:
                serp = np.append(serp, [serp_init2+x*toadd],axis=0)
                serp = np.append(serp, [serp_init1+x*toadd],axis=0)
        serp = serp[1::,:]

        # prune first and last
#         if not -1 <= prune_first <= 1:
#             raise ValueError('prune_first must be between -1 and 1.')
#         if not -1 <= prune_last <= 1:
#             raise ValueError('prune_last must be between -1 and 1.')        

        if left_right=='left':
            serp[0] = [serp[1][0]-(length-prune_first), serp[1][1]] 
        if left_right=='right':
            serp[0] = [serp[0][0]-prune_first, serp[0][1]] 
        if (left_right=='left' and np.mod(nbseg,2)==0 or 
                                 left_right=='right' and np.mod(nbseg,2)==1):
            serp[-1] = [serp[-2][0]-(length-prune_last), serp[-1][1]] 
        if (left_right=='left' and np.mod(nbseg,2)==1 or 
                                 left_right=='right' and np.mod(nbseg,2)==0):
            serp[-1] = [serp[-1][0]-prune_last, serp[-1][1]] 

        # rotate it to match orientation and direction
        if (left_right == 'right'):
            serp[:,0] = serp[:,0]-length

        if orientation == 'vertical':
            if (left_right == 'left') and (bottom_top == 'bottom'):
                serp = np.fliplr(serp)
            elif(left_right == 'right') and (bottom_top == 'top'):
                serp = np.fliplr(serp)
            else:
                serp = np.fliplr(serp)
                serp = -serp

        for x in range(serp.shape[0]):
            serp[x,:] = serp[x,:]+origin

        curv = curvature*np.ones(serp.shape[0])
        curv[0] = 0
        curv[-1]= 0

        serp_obj = Feature.define_tube(serp,curv,rad)
        #serp_obj = cls()
        #serp_obj.coord = [serp]
        return serp_obj
    
    @classmethod
    def circle(cls, radius, position, open_circle=False):

        """
        Generates punching pad feature.

        Parameters
        ----------
        radius : float 
            circle radius 
        position : float
            circle position

        Returns
        -------
        mf_feature
            Coordinates of opne polygon representing the circle

        """

        nb_points = 2*np.pi*radius/1
        points1 = radius*np.transpose(np.concatenate(([np.cos(2*np.pi*np.arange(0,nb_points+1)/nb_points)],[np.sin(2*np.pi*np.arange(0,nb_points+1)/nb_points)]),axis=0))
        
        for y in range(points1.shape[0]):
            points1[y,:]=points1[y,:]+position
                
        circle_obj = cls()
        circle_obj.coord = [points1]
        circle_obj.open = open_circle
        return circle_obj
    
    @classmethod
    def circular_punching(cls, nb_points,outer_rad, position):

        """
        Generates punching pad feature.

        Parameters
        ----------
        nb_points : int 
            Number of points composing the circular 
        dist : float
            Separation between segments
        rad : float
            Tube radius
        curvature : float
            Curvature of turns
        origin: 2D list
            Position of the "start" of the serpentine
        left_right : str
            serpentine starts 'left' or 'right'
        bottom_top : str
            serpentine starts 'top' or 'bottom'
        hor_vert : serpentine is 'horizontal' or 'vertical'

        Returns
        -------
        mf_feature
            Coordinates of polygon representing the tube

        """

        rad1= outer_rad
        rad2 = 0.1*rad1
        rad3 = 0.02*rad1
        points1 = rad1*np.transpose(np.concatenate(([np.cos(2*np.pi*np.arange(0,nb_points+1)/nb_points)],[np.sin(2*np.pi*np.arange(0,nb_points+1)/nb_points)]),axis=0))
        curv = 0*np.ones(points1.shape[0])
        curv[0] = 0
        curv[-1] = 0
        punching_circ = Feature.define_tube(points1,curv,rad3)
        points2 = rad2*np.transpose(np.concatenate(([np.cos(2*np.pi*np.arange(0,nb_points+1)/nb_points)],[np.sin(2*np.pi*np.arange(0,nb_points+1)/nb_points)]),axis=0))
        sectors = [np.concatenate(([points2[x,:]],[((rad1-rad3)/rad1)*points1[x,:]],[((rad1-rad3)/rad1)*points1[x+1,:]],[points2[x+1,:]]),axis=0) for x in range(0,nb_points,2)]

        punching = [punching_circ.coord[0],points2]

        for x in range(len(sectors)):
            punching.append(sectors[x])

        for x in range(len(punching)):
            for y in range(punching[x].shape[0]):
                punching[x][y,:]=punching[x][y,:]+position

        punching_obj = cls()
        punching_obj.coord = punching
        return punching_obj
 

    @classmethod
    def punching_dotted(cls, position, punch_rad, rad, dotlen):
        nb_points = round(2*np.pi*punch_rad)
        xy = [(position[0]+punch_rad*np.cos(2*np.pi*i/nb_points), 
               position[1]+punch_rad*np.sin(2*np.pi*i/nb_points)) for i in range(nb_points)]
        obj = Feature.define_tube_broken(xy, punch_rad, rad, dotlen)
        return obj


    @classmethod
    def channel_array(cls, length, num, space, space_series, widths, origin, subsampling=1):
        """
        Generates a channel array feature

        The channel array is composed of a series of channels series. Each channel series of 'num' channel has given width
        taken from the list 'widths'. The length of the channels is constant. 
        One can specify both the distances between channels within a series and between channel series. Additionally, one
        can skip every n'th channel by subsampling.

        Parameters
        ----------
        nbseg : float 
            Channel length
        num : int
            Number of channels in a series
        space : float
            Separation between channels
        space_series : float
            Separation between channel-series
        width : list
            channel witdhs for each series
        origin : float
            Position of the array (top-left)
        subsampling: int
            if >0 Use only every subsampling'th channel
            if <0 remove every -subsampling'th channel

        Returns
        -------
        mf_feature
            List of 2d numpy arrays specifying the position of each channel

        """
        if (subsampling == 0) or (subsampling == -1):
            raise ValueError('Subsampling cannot be 0 or -1')
            
        ch_array = [np.zeros((4,2)) for x in range(num*len(widths))]
        count = 0
        for i in range(len(widths)):
            for j in range(num):
                if ((subsampling>0) and (np.mod(j,subsampling) ==0)) or ((subsampling<0) and (np.mod(j,-subsampling) != 0)):
                    xpos = origin[0]+j*space+i*(space*num+space_series)
                    points = [[xpos,origin[1]],[xpos, origin[1]-length]]
                    ch_array[count] = Feature.define_tube(points,[0,0],widths[i]/2).coord[0]                    
                count = count+1

        ch_array_obj = cls()
        ch_array_obj.coord = ch_array
        ch_array_obj.params = {'length':length, 'num':num, 'space':space, 'space_series':space_series, 'widths':widths, 'origin':origin, 'subsampling':subsampling}
        return ch_array_obj

    @classmethod
    def numbering(cls, num, scale, pos, rotation=0, space_factor=1.2):
        """
        Creates a "polygon-number" in the form of a feature.

        The number is a set of polygons that can be added as a feature to a design. The number itself,
        its position, scale and rotation can be chosen. By default, the size of a number is 1 in height.
        The number is centered on the position pos. Dots can also be used.

        Parameters
        ----------
        num : float
            Number to be turned into feature
        scale : float
            Size of the number is scaled by scale
        pos : list
            Position of the number
        rotatio  : float
            Angle (radians) by which the number should be rotated.

        Returns
        -------
        feature
            feature for dxf design

        """
        numbers = {}
        numbers_rad = {}

        #define all the numbers as paths. Some of them cannot be traced as a single path and thus
        #are composed of several paths 
        numbers[1] = [[[0,-0.5],[0,0.5]]]
        numbers[2] = [[[0.35,-0.5],[-0.35,-0.5],[-0.35,0],[0.35,0.0],[0.35,0.5],[-0.35,0.5]]]
        numbers[3] = [[[-0.35,-0.5],[0.35,-0.5],[0.35,0.5],[-0.35,0.5]],[[-0.35,0],[0.35,0]]]
        numbers[4] = [[[0.35,-0.5],[0.35,0.5]],[[0.35,0],[-0.35,0],[-0.35,0.5]]]
        numbers[5] = [[[-0.35,-0.5],[0.35,-0.5],[0.35,0],[-0.35,0],[-0.35,0.5],[0.35,0.5]]]
        numbers[6] = [[[0.35,0.5],[-0.35,0.5],[-0.35,-0.5],[0.35,-0.5],[0.35,0],[-0.35,0]]]
        numbers[7] = [[[-0.35,0.5],[0.35,0.5],[0.35,-0.5]]]
        numbers[8] = [[[0.0,-0.5],[0.35,-0.5],[0.35,0],[-0.35,0],[-0.35,-0.5],[0.05,-0.5]],
                                [[0,0],[0.35,0],[0.35,0.5],[-0.35,0.5],[-0.35,0],[0.05,0]]]
        numbers[9] = [[[-0.35,-0.5],[0.35,-0.5],[0.35,0.5],[-0.35,0.5],[-0.35,0],[0.35,0]]]
        numbers[0] = [[[0,-0.5],[0.35,-0.5],[0.35,0.5],[-0.35,0.5],[-0.35,-0.5],[0.05,-0.5]]]
        numbers['dot'] = [[[-0.0005,-0.0005],[0.0005,-0.0005],[0.0005,0.0005],[-0.0005,0.0005]]]

        rad = 0.1
        for x in numbers:
            for i in range(len(numbers[x])):
                numbers[x][i] = np.array(numbers[x][i])

        #adjust some segments so that 1) polygons of different parts of a number are overlapping
        #2) numbers with holes like 8 have tiny breaks somewhere.
        for x in numbers:
            numbers_rad[x] = []
            for y in range(len(numbers[x])):
                for z in range(numbers[x][y].shape[0]-1):
                    p1 = numbers[x][y][z,:].copy()
                    p2 = numbers[x][y][z+1,:].copy()

                    if p1[1]==p2[1]:
                        if np.abs(p1[1])==0.5:
                            p1[1] = p1[1]-np.sign(p1[1])*rad
                            p2[1] = p2[1]-np.sign(p1[1])*rad
                        p1[0] = p1[0]+np.sign(p1[0])*rad
                        p2[0] = p2[0]+np.sign(p2[0])*rad

                        if ((np.abs(p1[1])==0 or p1[1]>0) and (x==8 or x==0 or x==6)) or ((np.abs(p1[1])==0) and (x==9)):
                            if p1[0]>0:
                                p1[0] = p1[0]-2.07*rad
                            else:
                                p2[0] = p2[0]-2.07*rad
                    else:
                        if p1[1]==0:
                            p1[1] = p1[1]-np.sign(p2[1])*rad
                        if p2[1]==0:
                            p2[1] = p2[1]-np.sign(p1[1])*rad

                    numbers_rad[x].append(Feature.define_tube((p1,p2),[0,0], rad).coord[0])

            #make each number a single polygon (without holes)
            number_int = MultiPolygon([Polygon([tuple(z) for z in y]) for y in numbers_rad[x]])
            solution = ops.cascaded_union(number_int)
            solution = [np.array(solution.exterior.coords)]

            #change the scale of each number
            numbers_rad[x] = [scale*np.array(x) for x in solution]

        #create the required number
        full_num = []
        num = str(num)
        len_num = len(num)
        if '.' in num:
            len_num = len_num-1

        if len_num==1:
            count = np.array([-scale*space_factor,0])
        else:
            count = np.array([-scale*((len_num+1)*space_factor)/2,0])
        for x in num:
            if x=='.':
                for y in numbers_rad['dot']:
                    y = np.array([k+count+scale*np.array([space_factor/2,-0.4]) for k in y])
                    full_num.append(y)
            else:
                count = count + np.array([scale*space_factor,0])
                for y in numbers_rad[int(x)]:
                    y = np.array([z + count for z in y])
                    full_num.append(y)

        #Rotate the number by alpha
        if rotation !=0:

            alpha = rotation
            R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])

            for i in range(len(full_num)):
                full_num[i] = np.squeeze([np.dot([x],R) 
                                                       for x in full_num[i]])
            
        #move the number to position pos
        for x in range(len(full_num)):
            full_num[x] = np.array([z+ np.array(pos) for z in full_num[x]])
        
        num_obj = cls()
        num_obj.coord = full_num
        return num_obj

    @classmethod
    def number_array(cls, scale, num, space, space_series, num_series, 
                     origin, subsampling, rotation = 0, values=None, thin = True):
        """
        Generates a number array feature

        This is usually used in conjunction with the channel array feature to number the channels. Hence the positionning 
        of numbers follows the same logic.

        Parameters
        ----------
        scale : float 
            Scale of the numbers
        num : int
            Number of numbers in a series
        space : float
            Separation between numbers
        space_series : float
            Separation between number-series
        num_series : int
            number of number-series
        origin : float
            Position of the array (top-left)
        subsampling : int
            Use only every subsampling'th number
        rotation : float
            Angle in radians by which to rotate the numbers

        Returns
        -------
        mf_feature
            List of 2d numpy arrays specifying the coordinates of each number

        """
        if values is None:
            nums = range(num)
        else:
            nums = [values[i%len(values)] for i in range(num)]               
        
        all_numbers = []
        for i in range(num_series):
            for j in range(num):
                xpos = origin[0]+j*space+i*(space*num+space_series)
                if np.mod(j,subsampling) ==0:
                    if thin:
                        cur_num = Feature.define_text([xpos,origin[1]],str(nums[j]), scale, rotation).coord
                    else:
                        cur_num = Feature.numbering(nums[j], scale, [xpos,origin[1]],rotation).coord
                    for x in cur_num:
                        all_numbers.append(x)
        all_numbers_obj = cls()
        all_numbers_obj.coord = all_numbers
        if thin:
            all_numbers_obj.open = True
        return all_numbers_obj
    
    @classmethod
    def patterned_region(cls, global_shape,channel_width,channel_separation):
        """
        Creates a grid-like pattern clipped to a chosen shape

        This creates a grid of channels with a chosen width and separation. The grid is then 
        cut ("clipped") to fit the shape of a chosen global_shape

        Parameters
        ----------
        global_shape : 2D array 
            Shape to use of clipping the grid
        channel_width : float
            WIdths of the channels
        channel_separation : distance between channels (horizonally and vertically)

        Returns
        -------
        feature
            feature for dxf design

        """
        
        global_shape = np.array(global_shape)

        minx = np.min(global_shape[:,0])
        maxx = np.max(global_shape[:,0])
        miny = np.min(global_shape[:,1])
        maxy = np.max(global_shape[:,1])

        clip = Polygon([tuple(z) for z in global_shape])

        #create vertical grid and calculate its intersection with global shape
        vert_grid = [Feature.define_tube([[x,miny],[x,maxy]],[0,0],channel_width/2).coord[0] for x in np.arange(minx,maxx,channel_separation)]
        vert_grid = MultiPolygon([Polygon([tuple(z) for z in y]) for y in vert_grid])

        pattern = []

        intersection = vert_grid.intersection(clip)
        if intersection.geom_type == 'Polygon':
            pattern.append(np.array(intersection.exterior.coords))
        if intersection.geom_type == 'MultiPolygon':
            for x in intersection:
                pattern.append(np.array(x.exterior.coords))

        #create horizontal grid and calculate its intersection with global shape
        hor_grid = [Feature.define_tube([[minx,y],[maxx,y]],[0,0],channel_width/2).coord[0] for y in np.arange(miny,maxy,channel_separation)]
        hor_grid = MultiPolygon([Polygon([tuple(z) for z in y]) for y in hor_grid])

        intersection = hor_grid.intersection(clip)
        if intersection.geom_type == 'Polygon':
            pattern.append(np.array(intersection.exterior.coords))
        if intersection.geom_type == 'MultiPolygon':
            for x in intersection:
                pattern.append(np.array(x.exterior.coords))

        pattern_obj = cls()
        pattern_obj.coord = pattern
        return pattern_obj
    
    @classmethod
    def align_mark_squares(cls, pos, rotation = False, 
                           overlap = 2, sq_sizes = [4, 7, 12, 22, 32, 132, 202]):

        """
        Creates standard alignement marks for SU8 lithography

        Creates a pair of alignement marks in the form of two features. One can set their position
        as well as their orientation (it's useful to use both regular and 45° rotated version to have
        more precision and/or use multilayers).

        Parameters
        ----------
        pos : 2D list 
            Position of the mark
        rotation  : boolean
            False do not rotate, True rotate by 45°

        Returns
        -------
        tuple of features
            features for dxf design

        """

        pos_init = np.array([-overlap/2,overlap/2])
        all_sq = []

        init_sq = np.array([[0,0], [0,sq_sizes[0]],[-sq_sizes[0],sq_sizes[0]],[-sq_sizes[0],0]])
        init_sq = np.array([np.array([-overlap/2,overlap/2])+x for x in init_sq])

        all_sq.append(init_sq)
        all_sq.append(np.stack((-init_sq[:,0],init_sq[:,1]),axis=1))
        all_sq.append(np.stack((init_sq[:,0],-init_sq[:,1]),axis=1))
        all_sq.append(np.stack((-init_sq[:,0],-init_sq[:,1]),axis=1))

        for i in range(1,len(sq_sizes)):
            pos_init = pos_init-np.array([(sq_sizes[i-1]-overlap),-(sq_sizes[i-1]-overlap)])
            init_sq = np.array([[0,0], [0,sq_sizes[i]],[-sq_sizes[i],sq_sizes[i]],[-sq_sizes[i],0]])
            init_sq = np.array([pos_init+x for x in init_sq])

            all_sq.append(init_sq)
            all_sq.append(np.stack((-init_sq[:,0],init_sq[:,1]),axis=1))
            all_sq.append(np.stack((init_sq[:,0],-init_sq[:,1]),axis=1))
            all_sq.append(np.stack((-init_sq[:,0],-init_sq[:,1]),axis=1))

        #create complementary features
        all_sq2 = []
        for i in range(len(sq_sizes)):

            if i==0:
                overlap_loc = -overlap/2;
            else:
                overlap_loc = overlap;

            it = i*4;
            init_sq = np.array([[all_sq[it][0,0],all_sq[it][0,1]+overlap_loc],
                                [all_sq[it+1][0,0],all_sq[it+1][0,1]+overlap_loc],
                                [all_sq[it+1][0,0],all_sq[it+1][1,1]],
                                [all_sq[it][0,0],all_sq[it][1,1]]])
            all_sq2.append(init_sq)

            init_sq = np.array([[all_sq[it+2][0,0],all_sq[it+2][0,1]-overlap_loc],
                                        [all_sq[it+3][0,0],all_sq[it+3][0,1]-overlap_loc],
                                        [all_sq[it+3][0,0],all_sq[it+3][1,1]],
                                        [all_sq[it+2][0,0],all_sq[it+2][1,1]]])
            all_sq2.append(init_sq)

            init_sq = np.array([[all_sq[it][0,0]-overlap_loc,all_sq[it][0,1]],
                                [all_sq[it][3,0],all_sq[it][3,1]],
                                [all_sq[it+2][3,0],all_sq[it+2][3,1]],
                                [all_sq[it+2][0,0]-overlap_loc,all_sq[it+2][0,1]]])
            all_sq2.append(init_sq)

            init_sq = np.array([[all_sq[it+1][0,0]+overlap_loc,all_sq[it+1][0,1]],
                                [all_sq[it+1][3,0],all_sq[it+1][3,1]],
                                [all_sq[it+3][3,0],all_sq[it+3][3,1]],
                                [all_sq[it+3][0,0]+overlap_loc,all_sq[it+3][0,1]]])
            all_sq2.append(init_sq)

        #rotate marks
        if rotation:
            alpha = np.pi/4
            R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])

            for i in range(len(all_sq)):
                all_sq[i] = np.squeeze([np.dot([x + np.array([overlap/2,-overlap/2])],R) for x in all_sq[i]])
                all_sq2[i] = np.squeeze([np.dot([x + np.array([overlap/2,-overlap/2])],R) for x in all_sq2[i]])

        pos = np.array(pos)
        all_sq = [np.array([y+pos for y in x]) for x in all_sq]
        all_sq2 = [np.array([y+pos for y in x]) for x in all_sq2]

        align1_obj = cls()
        align2_obj = cls()
        align1_obj.coord = all_sq
        align2_obj.coord = all_sq2
    
        return (align1_obj, align2_obj)
    
    @classmethod
    def align_mark_rulers(cls, pos, rotation = False, 
                           half_length = 250, thickness = 10, cross_thickness = 50, 
                           pitches = [500, 100, 25], widths = [100, 75, 50]):

        """
        Creates standard alignement marks for SU8 lithography

        Creates a pair of alignement marks in the form of two features. One can set their position
        as well as their orientation (it's useful to use both regular and 45° rotated version to have
        more precision and/or use multilayers).

        Parameters
        ----------
        pos : 2D list 
            Position of the mark
        rotation  : boolean
            False do not rotate, True rotate by 90°

        Returns
        -------
        tuple of features
            features for dxf design

        """

        horiz = cls()
        horiz.coord=[]
        for i in range(len(widths)):
            horiz = Feature.combine_features(horiz, 
                Feature.channel_array(widths[i], half_length // pitches[i] + 1, pitches[i], 0, [thickness], 
                                        [0, -(2 * half_length - max(widths)) ]) )
        horiz = Feature.combine_features(horiz, horiz.mirror_feature(x=0) )

        vert = horiz.copy()
        vert.rotate([2 * half_length - max(widths), -(2 * half_length - max(widths))], np.pi/2)
        
        cross = Feature.combine_features(
            Feature.define_tube(([-half_length, 0], [half_length, 0]), 0, cross_thickness), 
            Feature.define_tube(([0, -half_length], [0, half_length]), 0, cross_thickness) )

        align1_obj = Feature.combine_features(horiz, vert)
        align1_obj = Feature.combine_features(align1_obj, cross)
        align2_obj = Feature.combine_features(
            horiz.mirror_feature(y=-(2 * half_length - max(widths)) + (min(pitches)-thickness)/2), 
            vert.mirror_feature(x=2 * half_length - max(widths) + (min(pitches)-thickness)/2) )
        align2_obj = Feature.combine_features(align2_obj, cross)
        
        if rotation:
            align1_obj.rotate([0,0], np.pi/2)
            align2_obj.rotate([0,0], np.pi/2)
        align1_obj.move(pos)
        align2_obj.move(pos)
        
        return (align1_obj, align2_obj)
    
    @classmethod
    def pad_with_filter(cls, position, filter_size, filter_number, pad_size, rect_size, funnel_width, funnel_end):

        """
        Creates a punching pad with filter.

        The filter region is composed of arrays of squares of different sizes. The size of 
        the filter region is set by the number and sizes of those arrays. The filter region is flanked on one side 
        by a half-circle punching region and on the other by a funnel-shaped region that connnects to a channel. 

        Parameters
        ----------
        position : 2D list 
            Position of pad. This point locates the point where the pad is connected to a potential channel
        filter_size  : 2D list 
            Sizes of squares forming the filter
        filter_number : 2D list
            Number of series of each square size
        pad_size : float
            Height of the pad. Also sets the diameter of half-circle punching pad
        rect_size : float
            Width of the rectangular region between hafl-circle and filers
        funnel_width : float
            Width of funnel-shaped region
        funnel_end : float
            height of the smaller section of the funnel that connects to a potential channel

        Returns
        -------
        feature

        Usage
        -----
        filterpad_feature = Feature.pad_with_filter([0,0], [20,10,5], [2,2,5], 500, 50, 100, 10)
        """
        
        position = np.array(position)
        filter_size = np.array(filter_size)
        filter_number = np.array(filter_number)
        
        #with of rectangle between half-circle and filters
        init_square = rect_size

        #lower-left position of the first "filter-channel
        init_pos = np.array([position[0]-np.sum(filter_size*filter_number*2)-funnel_width,position[1]-pad_size/2])

        #pattern contains all coordinates composing the filter
        #first define the funnel part connecting to a channel
        pattern = [np.array([[position[0]-funnel_width,position[1]-pad_size/2],[position[0]-funnel_width,position[1]+pad_size/2],
                             [position[0],position[1]+funnel_end/2],[position[0],position[1]-funnel_end/2]])]

        #define rectangular part between half-circle and filter
        pattern.append(np.array([[init_pos[0]-init_square,init_pos[1]],[init_pos[0],init_pos[1]],
                                 [init_pos[0],init_pos[1]+pad_size],[init_pos[0]-init_square,init_pos[1]+pad_size]]))

        #define the half-circle punching pad
        half_circle = np.array([[init_pos[0]-init_square+pad_size/2*np.cos(x),position[1]+pad_size/2*np.sin(x)] 
                                for x in np.linspace(np.pi/2,np.pi*3/2)])
        pattern.append(half_circle)

        #define filter region composed of vertical channels and horizontal squares
        for i in range(len(filter_size)):
            for f in range(filter_number[i]):

                shift = np.mod(f,2)*filter_size[i]
                sq_coord = [[init_pos[0]+filter_size[i],init_pos[1]],[init_pos[0]+2*filter_size[i],init_pos[1]],
                                [init_pos[0]+2*filter_size[i],init_pos[1]+pad_size],[init_pos[0]+filter_size[i],init_pos[1]+pad_size]]
                pattern.append(np.array(sq_coord))

                for j in range(int(np.round(pad_size/filter_size[i]/2))):

                    sq_coord = [[init_pos[0],init_pos[1]+shift],[init_pos[0]+filter_size[i],init_pos[1]+shift],
                                [init_pos[0]+filter_size[i],init_pos[1]+filter_size[i]+shift],[init_pos[0],init_pos[1]+filter_size[i]+shift]]
                    pattern.append(np.array(sq_coord))
                    init_pos = init_pos+np.array([0,2*filter_size[i]])
                init_pos[0]=init_pos[0]+2*filter_size[i]
                init_pos[1]=position[1]-pad_size/2
        
        filter_pad_obj = cls()
        filter_pad_obj.coord = pattern
    
        return filter_pad_obj

    @classmethod
    def inline_filter(cls, position, pore_size, pore_number, filter_size, funnel_width, funnel_tip, rad, before=0, after=0, pore_dist=None):

        """
        Creates a punching pad with filter.

        The filter region is composed of arrays of squares of different sizes. The size of 
        the filter region is set by the number and sizes of those arrays. The filter region is flanked on one side 
        by a half-circle punching region and on the other by a funnel-shaped region that connnects to a channel. 

        Parameters
        ----------
        position : 2D list 
            Position of pad. This point locates the point where the filter output (excluding a possible tube after)
        pore_size  : 2D list 
            Sizes of squares forming the filter
        pore_number : 2D list
            Number of series of each square size
        filter_size : float
            Height of the filter. 
        funnel_width : float
            Width of funnel-shaped region
        rad : float
            half height of the smaller section of the funnel that connects to a potential channel

        Returns
        -------
        feature

        Usage
        -----
        filter_feature = Feature.inline_filter([0,0], [20,10,5], [2,2,5], 500, 50, 50, 25, 0, 0)
        """
        
        position = np.array(position)
        pore_size = np.array(pore_size)
        pore_number = np.array(pore_number)
        if pore_dist is None:
            pore_dist = np.ones(len(pore_size))
        else:
            pore_dist = np.array(pore_dist)
        #with of rectangle between half-circle and filters
#        init_square = rect_size

        #lower-left position of the first "filter-channel
        init_pos = np.array([position[0]-np.sum(pore_size*pore_number*2)-funnel_width,position[1]-filter_size/2])

        #pattern contains all coordinates composing the filter
        #first define the output funnel part
        pattern = [np.array([[position[0]-funnel_width,position[1]-filter_size/2],
                             [position[0]-funnel_width,position[1]+filter_size/2],
                             [position[0],position[1]+funnel_tip],[position[0],position[1]-funnel_tip]])]

        #define rectangular part between half-circle and filter
        pattern.append(np.array([[init_pos[0]-pore_size[0],init_pos[1]],[init_pos[0],init_pos[1]],
                                 [init_pos[0],init_pos[1]+filter_size],[init_pos[0]-pore_size[0],init_pos[1]+filter_size]]))

        # define the input funnel part
        pattern.append(np.array([[init_pos[0]-pore_size[0],init_pos[1]], 
                                 [init_pos[0]-pore_size[0],init_pos[1]+filter_size],
                                 [init_pos[0]-pore_size[0]-funnel_width,position[1]+funnel_tip], 
                                 [init_pos[0]-pore_size[0]-funnel_width,position[1]-funnel_tip]])) 
        
        # append tubes before and after
        if before > 0:
            pattern.append(np.array([[init_pos[0]-pore_size[0]-funnel_width,position[1]+rad], 
                                     [init_pos[0]-pore_size[0]-funnel_width,position[1]-rad],
                                     [init_pos[0]-pore_size[0]-funnel_width-before,position[1]-rad], 
                                     [init_pos[0]-pore_size[0]-funnel_width-before,position[1]+rad] ])) 
        if after > 0:
            pattern.append(np.array([[position[0],position[1]+rad],[position[0],position[1]-rad],
                                     [position[0]+after,position[1]-rad],[position[0]+after,position[1]+rad]])) 

        #define filter region composed of vertical channels and horizontal squares
        for i in range(len(pore_size)):
            for f in range(pore_number[i]):
                sq_coord = [[init_pos[0]+pore_size[i],init_pos[1]],[init_pos[0]+2*pore_size[i],init_pos[1]],
                                [init_pos[0]+2*pore_size[i],init_pos[1]+filter_size],[init_pos[0]+pore_size[i],init_pos[1]+filter_size]]
                pattern.append(np.array(sq_coord))

                shift = np.mod(f,2)*np.round(pore_size[i]*(1+pore_dist[i])/2)
                for j in range(int(np.round(filter_size/(pore_size[i]*(1+pore_dist[i]))))):
                    sq_coord = [[init_pos[0],init_pos[1]+shift],[init_pos[0]+pore_size[i],init_pos[1]+shift],
                                [init_pos[0]+pore_size[i],init_pos[1]+pore_size[i]+shift],[init_pos[0],init_pos[1]+pore_size[i]+shift]]
                    pattern.append(np.array(sq_coord))
                    init_pos = init_pos+np.array([0,pore_size[i]*(1+pore_dist[i])])
                init_pos[0]=init_pos[0]+pore_size[i]*2
                init_pos[1]=position[1]-filter_size/2
        
        filter_pad_obj = cls()
        filter_pad_obj.coord = pattern
    
        return filter_pad_obj


    
        
    def mirror_feature(self, x=None, y=None):
        """
        Return a flipped copy of a feature along the vertical axis located at position x
        and along the horizontal axis located at position y.

        Parameters
        ----------
        x : float
            horizontal position of the vertical axis
        y : float
            vertical position of the horizontal axis

        Returns
        -------
        feature
            Mirrored feature

        """

        #mirrored = Feature()
        mirrored = copy.deepcopy(self)
        if not x == None:
            for i in range(mirrored.feature_len()):
                mirrored.coord[i] = np.array([[2*x-c[0],c[1]] for c in mirrored.coord[i]])
        if not y == None:
            for i in range(mirrored.feature_len()):
                mirrored.coord[i] = np.array([[c[0],2*y-c[1]] for c in mirrored.coord[i]])        
        return mirrored

    def feature_len(self):
        """
        Returns the number of parts composing a feature.

        Parameters
        ----------
        
        Returns
        -------
        int
            number of parts within a feature

        """
        return len(self.coord)
    
    def copy(self):
        """
        Returns a copy of a feature.
        
        This makes a deep copy to ensure features are trully independent.

        Parameters
        ----------
        
        Returns
        -------
        feature
            copied feature

        """
        return copy.deepcopy(self)
    
    def move(self,move):
        """
        Moves a feature.
        
        Adds to each coordinate of a feature the displacement given by the parameter move.

        Parameters
        ----------
        
        move: 2-element list
            displacement to add to a feature
        
        Returns
        -------
        feature
            feature moved by the displacement move

        """
        for x in range(len(self.coord)):
            self.coord[x] = np.array([y+np.array(move) for y in self.coord[x]])
        return self
    
    def rotate(self,center, angle):
        """
        Moves a feature.
        
        Adds to each coordinate of a feature the displacement given by the parameter move.

        Parameters
        ----------
        
        center: 2D list
            center of rotation
        angel: float
            angle of rotation
        
        Returns
        -------
        feature
            feature roated by angle around center

        """
        
        self.coord = [x-np.repeat([[center[0],center[1]]],[x.shape[0]],axis = 0) for x in self.coord]

        alpha = angle
        R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
        
        for i in range(len(self.coord)):
            self.coord[i] = np.squeeze([np.dot([x],R) for x in self.coord[i]])

        self.coord = [x+np.repeat([[center[0],center[1]]],[x.shape[0]],axis = 0) for x in self.coord]

        return self
    
    
    def combine_features(feature1, feature2):
        """
        Combined two features into single feature
        
        Creates a new feature by aggregating all the parts of two features into a single feature.

        Parameters
        ----------
        
        feature1: feature
        featzre2: feature
        
        Returns
        -------
        feature
            new feature aggregating two input features

        """
        new_feature = Feature()
        new_feature.coord = feature1.coord.copy()
        for x in feature2.coord:
            new_feature.coord.append(x)
        return new_feature
    

    def reverse_feature(feature, back_square):
        """
        Reverses a feature from polygons to hole within a rectangle.

        Transforms each polygon of a feature into a hole within a given rectangle. Of course the features and 
        the rectangle have to be overlapping for holes to be created.

        Parameters
        ----------
        feature : feature
            feature, can contain any element or mix of elements
        back_square : 2D array
            Coordinates of the rectangle in which the features should appear as holes

        Returns
        -------
        feature
            original rectangle with holes

        """
        back_square = np.array(back_square)
        feature1 = MultiPolygon([Polygon([tuple(z) for z in y]) for y in feature.coord])

        init = np.min(back_square[:,1])
        height =0.5
        back = []

        while (init + height<=np.max(back_square[:,1])):

            test_square = np.array([[np.min(back_square[:,0]),init],[np.max(back_square[:,0]),init],
                                    [np.max(back_square[:,0]),init+height],[np.min(back_square[:,0]),init+height]])
            feature2 = Polygon([tuple(z) for z in test_square])
            difference = feature2.difference(feature1)
            newdifference = difference

            while (has_hole(newdifference)==0) and (init + height<=np.max(back_square[:,1])):
                difference = newdifference
                height = height+0.5
                test_square = np.array([[np.min(back_square[:,0]),init],[np.max(back_square[:,0]),init],
                                    [np.max(back_square[:,0]),init+height],[np.min(back_square[:,0]),init+height]])
                feature2 = Polygon([tuple(z) for z in test_square])
                newdifference = feature2.difference(feature1)
            if difference.geom_type == 'Polygon':
                back.append(np.array(difference.exterior.coords))
            if difference.geom_type == 'MultiPolygon':
                for x in difference:
                    back.append(np.array(x.exterior.coords))
            init = init+height-0.5
            height = 0.5
            
        topoinvert_obj = Feature()
        topoinvert_obj.coord = back
        return topoinvert_obj
    
    def channel_array_blocks(self, opening_width, block_len, block_from_bottom):
        """
        Creates blocks within channels of a channel array.


        Parameters
        ----------
        channel_array : feature
            channel_array to be modfied
        opening_width : float
            free space on each side of the block
        block_len : float
            height of the block
        block_from_bottom: float
            distance of the block from the bottom of the channel

        Returns
        -------
        feature
            modified channel array

        """
        
        params = self.params
        count = 0
        for i in range(len(params['widths'])):
            if params['subsampling']>0:
                back_square = self.coord[i*params['num']]
            else:
                back_square = self.coord[i*params['num']+1].copy()
                back_square = back_square-np.repeat([[params['space'],0]],[back_square.shape[0]],axis = 0)
                
            center_x = 0.5*(np.min(back_square[:,0])+np.max(back_square[:,0]))
            center_y = np.min(back_square[:,1])                     
            block = Feature.define_polygon([[center_x-params['widths'][i]/2+opening_width,center_y+block_from_bottom],[center_x+params['widths'][i]/2-opening_width,center_y+block_from_bottom],[center_x+params['widths'][i]/2-opening_width,center_y+block_from_bottom+block_len],[center_x-params['widths'][i]/2+opening_width, center_y+block_from_bottom+block_len]])
                                   
            temp = Feature.reverse_feature(block, back_square)
            for j in range(params['num']):
                if ((params['subsampling']>0) and (np.mod(j,params['subsampling']) ==0)) or ((params['subsampling']<0) and (np.mod(j,-params['subsampling']) != 0)):
                    new_coord = temp.coord
                    new_coord = [x+np.repeat([[j*params['space'],0]],[x.shape[0]],axis = 0) for x in new_coord]
                    self.coord[count] = new_coord
                count+=1
        #self.coord = [item for sublist in self.coord for item in sublist]
        temp = []
        for x in self.coord:
            if(isinstance(x, list)):
                for y in x: 
                    temp.append(y)
            else:
                temp.append(x)
        self.coord = temp

        
        
        '''params = self.params
        myarray2 = Feature.channel_array(length=block_len,num=params['num'],space = params['space'],space_series = params['space_series'],widths = [x-2*opening_width for x in params['widths']],origin=np.array(params['origin'])+np.array([0,-params['length']+block_len+block_from_bottom]), subsampling=params['subsampling'])
        new_feature = Feature()
        for i in range(len(self.coord)):
            back_square = self.coord[i]
            curr_feature = Feature()
            curr_feature.coord = [myarray2.coord[i]]

            temp = Feature.reverse_feature(curr_feature, back_square)
            if new_feature.coord:
                new_feature = Feature.combine_features(new_feature,temp)
            else:
                new_feature = temp
        self.coord = new_feature.coord'''
        return self
    
    
def has_hole(feature):
    """
    Detects the number of holes in a shapely polygon or multipolygon.

    Parameters
    ----------
    feature : shapely Polygon or Multipolygon
        polygon to be analyzed for holes
            
    Returns
    -------
    int
        number of holes

    """
    if feature.geom_type == 'Polygon':
        num_holes = len(feature.interiors)
    elif feature.geom_type == 'MultiPolygon':
        num_holes = np.sum([len(x.interiors) for x in feature])
    return num_holes




