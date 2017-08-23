import numpy as np
import matplotlib.pyplot as plt
from dxfwrite import DXFEngine as dxf

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely import ops

import copy

class Design:
    
    def __init__(self):
        
        self.features = {}
        self.layers = {}
        self.file = None
        #Feature.design = self
        
    
    def __add__(self, other):
        sum_design = copy.deepcopy(self)
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
        
        
    def add_layer(self, layer_name, layer):
        self.layers[layer_name] = layer
        
    def add_feature(self, name, feature):
        self.features[name] = feature
        
        if not feature.mirror == None:
            self.features[name+'_mirror'] = feature.flip_feature(feature.mirror)
                
    def get_coord(self,name):
        return self.features[name].coord
    
    def add_to_coord(self,name,item,toadd):
        self.features[name].coord[item] = np.array([x+np.array(toadd) for x in self.features[name].coord[item]])
        
    def multiplicate(self, positions):
        complete_design = Design()
        complete_design.layers = self.layers
        complete_design.file = self.file
        
        counter = 0
        for p in positions:
            original = copy.deepcopy(self)
            original.set_design_origin(p)
            for f in original.features:
                original.features[f].mirror = None
                complete_design.add_feature(f+str(counter),original.features[f])
            counter  = counter +1
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

    def add_closed_polyline(self, layer_to_use,poly,drawing):
        """
        Adds a feature to the DXF drawing.

        Adds a feature (numpy polygon or list of numpy polygons) in a chosen layer 
        of a DXF drawing opened using the dxfwrite package.

        Parameters
        ----------
        layer_to_use : dictionary 
            dictionary with field 'name' of the layer 
        poly : feature
            Featue to write in file
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
            #self.add_closed_polyline(self.features[x].layer,self.features[x].coord,drawing)
            self.add_closed_polyline(self.layers[self.features[x].layer],self.features[x].coord,drawing)        
            #if not self.features[x].mirror == None:
                #self.add_closed_polyline(self.layers[self.features[x].layer],self.features[x].flip_feature(self.features[x].mirror).coord,drawing)
        self.drawing = drawing
        #return drawing
        
    def close(self):
        self.drawing.save()
        
    
    
class Feature:
    
    #design = None
    
    def __init__(self, coord = None, layer=None, mirror=None):
        
        self.coord = coord
        self.layer = layer
        self.mirror = mirror
        
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
        
        
        complete = np.array([points[0,:]])
        for i in range(1,len(curvature)):
            if curvature[i] != 0:
                vec1 = (points[i+1,:]-points[i,:])/np.linalg.norm(points[i+1,:]-points[i,:])
                vec2 = (points[i-1,:]-points[i,:])/np.linalg.norm(points[i-1,:]-points[i,:])
                
                if vec1[0]*vec2[1]-vec1[1]*vec2[0] == 0:
                    complete = np.append(complete,[points[i,:]],axis=0)
                    continue
                
                bis = (vec1+vec2)/np.linalg.norm(vec1+vec2)

                gamma = np.arccos(np.dot(vec1,vec2))/2
                D = curvature[i]*np.tan(np.pi/2-gamma)
                D2 = np.sqrt(curvature[i]**2+D**2)
                alpha2 = np.pi/2-gamma


                if np.cross(vec1,vec2)==0:
                    complete = np.append(complete,points[i,:],axis=0)
                    continue

                if np.cross(vec1,vec2)<0:
                    angles = np.arange(-alpha2,alpha2,0.1)
                else:
                    angles = np.arange(alpha2,-alpha2,-0.1)

                center = points[i,:]+D2*bis

                vect = -curvature[i]*bis
                arc = np.squeeze([center+np.dot(vect,np.array([[np.cos(x),-np.sin(x)],[np.sin(x),np.cos(x)]])) for x in angles])
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
        prune_first : float (optionnal)
            cut a given fraction of the first segment
        prune_last : float (optionnal)
            cut a given fraction of the last segment


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
        if not 0 <= prune_first <= 1:
            raise ValueError('prune_first must be between 0 and 1.')
        if not 0 <= prune_last <= 1:
            raise ValueError('prune_last must be between 0 and 1.')        

        if prune_first>0 and left_right=='left':
            serp[0] = [serp[1][0]*prune_first, serp[1][1]] 
        if prune_first>0 and left_right=='right':
            serp[0] = [serp[0][0]*(1-prune_first), serp[0][1]] 
        if prune_last>0 and (left_right=='left' and np.mod(nbseg,2)==0 or 
                                 left_right=='right' and np.mod(nbseg,2)==1):
            serp[-1] = [serp[-2][0]*prune_last, serp[-1][1]] 
        if prune_last>0 and (left_right=='left' and np.mod(nbseg,2)==1 or 
                                 left_right=='right' and np.mod(nbseg,2)==0):
            serp[-1] = [serp[-1][0]*(1-prune_last), serp[-1][1]] 

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
    def channel_array(cls, length, num, space, space_series, widths, origin, subsampling):
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
            Use only every subsampling'th channel

        Returns
        -------
        mf_feature
            List of 2d numpy arrays specifying the position of each channel

        """
        ch_array = [np.zeros((4,2)) for x in range(num*len(widths))]
        count = 0
        for i in range(len(widths)):
            for j in range(num):
                if np.mod(j,subsampling) ==0:
                    xpos = origin[0]+j*space+i*(space*num+space_series)
                    points = [[xpos,origin[1]],[xpos, origin[1]-length]]
                    ch_array[count] = Feature.define_tube(points,[0,0],widths[i]/2).coord[0]
                    count = count+1

        ch_array_obj = cls()
        ch_array_obj.coord = ch_array
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
    def number_array(cls, scale, num, space, space_series, num_series, origin, subsampling, rotation = 0, values=None):
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
                    cur_num = Feature.numbering(nums[j], scale, [xpos,origin[1]],rotation).coord
                    for x in cur_num:
                        all_numbers.append(x)
        all_numbers_obj = cls()
        all_numbers_obj.coord = all_numbers
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
    def align_mark_squares(cls, pos, rotation = False):

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

        overlap = 2
        sq_sizes = [4, 7, 12, 22, 32, 132, 202]
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
    
        
    def flip_feature(self, ax):
        """
        Flips a feature along the horizontal axis located at position ax.

        Parameters
        ----------
        ax : float
            vertical position of the horizontal axis

        Returns
        -------
        feature
            Mirrored feature

        """

        #mirrored = Feature()
        mirrored = copy.deepcopy(self)
        for i in range(mirrored.feature_len()):
            mirrored.coord[i] = np.array([[x[0],2*ax-x[1]] for x in mirrored.coord[i]])
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