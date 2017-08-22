import numpy as np
import matplotlib.pyplot as plt
from dxfwrite import DXFEngine as dxf

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely import ops

def draw_design(design, drawing):
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
    for x in design:
        add_closed_polyline(design[x]['layer'],design[x]['coord'],drawing)
        if 'mirror' in design[x]:
            add_closed_polyline(design[x]['layer'],flip_feature(design[x]['coord'],design[x]['mirror']),drawing)
    return drawing

def add_closed_polyline(layer_to_use,poly,drawing):
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


def set_design_origin(design,origin):
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
    for d in design:
        if type(design[d]['coord']) is list:
            for i in range(len(design[d]['coord'])):
                design[d]['coord'][i] = np.array([[x[0]+origin[0],x[1]+origin[1]] for x in design[d]['coord'][i]])
        else:
            design[d]['coord'] = np.array([[x[0]+origin[0],x[1]+origin[1]] for x in design[d]['coord']])
    return design


def define_tube(points,curvature, rad):
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
    #plt.plot(tube[:,0], tube[:,1],'-')
    #plt.axis('equal')
    #plt.show()
    
    return tube

def flip_feature(feature,ax):
    
    """
    Flips a feature along the horizontal axis located at position ax.

    Parameters
    ----------
    feature : 2D numpy array or list of 2D numpy array
        Design feature
    ax : float
        vertical position of the horizontal axis

    Returns
    -------
    feature
        Mirrored feature

    """
    
    if type(feature) is list:
        mirrored = feature
        for i in range(len(mirrored)):
            mirrored[i] = np.array([[x[0],2*ax-x[1]] for x in mirrored[i]])
    else:
        mirrored = np.array([[x[0],2*ax-x[1]] for x in feature])
    return mirrored

def serpentine(nbseg, dist, rad, length, curvature, origin, left_right,bottom_top, hor_vert):
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
    
    if (left_right == 'right'):
        serp[:,0] = serp[:,0]-length
    
    if hor_vert == 'vertical':
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

    serp = define_tube(serp,curv,rad)
    return serp

def circular_punching(nb_points,outer_rad, position):
    
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
    punching_circ = define_tube(points1,curv,rad3)
    points2 = rad2*np.transpose(np.concatenate(([np.cos(2*np.pi*np.arange(0,nb_points+1)/nb_points)],[np.sin(2*np.pi*np.arange(0,nb_points+1)/nb_points)]),axis=0))
    sectors = [np.concatenate(([points2[x,:]],[((rad1-rad3)/rad1)*points1[x,:]],[((rad1-rad3)/rad1)*points1[x+1,:]],[points2[x+1,:]]),axis=0) for x in range(0,nb_points,2)]

    punching = [punching_circ,points2]

    for x in range(len(sectors)):
        punching.append(sectors[x])

    for x in range(len(punching)):
        for y in range(punching[x].shape[0]):
            punching[x][y,:]=punching[x][y,:]+position
    
    return punching


def channel_array(length, num, space, space_series, widths, origin, subsampling):
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
                ch_array[count] = define_tube(points,[0,0],widths[i]/2)
                count = count+1
        
    return ch_array

def number_array(scale, num, space, space_series, num_series, origin, subsampling, rotation = 0):
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
    all_numbers = {}
    all_numbers['numbers'] = []
    #all_numbers['envelopes'] = []
    for i in range(num_series):
        for j in range(num):
            xpos = origin[0]+j*space+i*(space*num+space_series)
            if np.mod(j,subsampling) ==0:
                #cur_num = numbering(j,scale,[xpos,origin[1]])
                cur_num = numbering(j, scale, [xpos,origin[1]],rotation)
                for x in cur_num:#['numbers']:
                    all_numbers['numbers'].append(x)
                #for x in cur_num['envelopes']:
                #    all_numbers['envelopes'].append(x)
    return all_numbers['numbers']
            

def patterned_region(global_shape,channel_width,channel_separation):
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
    vert_grid = [define_tube([[x,miny],[x,maxy]],[0,0],channel_width/2) for x in np.arange(minx,maxx,channel_separation)]
    vert_grid = MultiPolygon([Polygon([tuple(z) for z in y]) for y in vert_grid])

    pattern = []
    
    intersection = vert_grid.intersection(clip)
    if intersection.geom_type == 'Polygon':
        pattern.append(np.array(intersection.exterior.coords))
    if intersection.geom_type == 'MultiPolygon':
        for x in intersection:
            pattern.append(np.array(x.exterior.coords))
       
    #create horizontal grid and calculate its intersection with global shape
    hor_grid = [define_tube([[minx,y],[maxx,y]],[0,0],channel_width/2) for y in np.arange(miny,maxy,channel_separation)]
    hor_grid = MultiPolygon([Polygon([tuple(z) for z in y]) for y in hor_grid])
    
    intersection = hor_grid.intersection(clip)
    if intersection.geom_type == 'Polygon':
        pattern.append(np.array(intersection.exterior.coords))
    if intersection.geom_type == 'MultiPolygon':
        for x in intersection:
            pattern.append(np.array(x.exterior.coords))
    
    return pattern


def align_mark_squares(pos, rotation = False):

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
    
    return (all_sq, all_sq2)


def numbering(num, scale, pos, rotation=0):
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
    numbers[2] = [[[0.5,-0.5],[-0.5,-0.5],[-0.5,0],[0.5,0.0],[0.5,0.5],[-0.5,0.5]]]
    numbers[3] = [[[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]],[[-0.5,0],[0.5,0]]]
    numbers[4] = [[[0.5,-0.5],[0.5,0.5]],[[0.5,0],[-0.5,0],[-0.5,0.5]]]
    numbers[5] = [[[-0.5,-0.5],[0.5,-0.5],[0.5,0],[-0.5,0],[-0.5,0.5],[0.5,0.5]]]
    numbers[6] = [[[0.5,0.5],[-0.5,0.5],[-0.5,-0.5],[0.5,-0.5],[0.5,0],[-0.5,0]]]
    numbers[7] = [[[-0.5,0.5],[0.5,0.5],[0.5,-0.5]]]
    numbers[8] = [[[0.0,-0.5],[0.5,-0.5],[0.5,0],[-0.5,0],[-0.5,-0.5],[0.05,-0.5]],
                            [[0,0],[0.5,0],[0.5,0.5],[-0.5,0.5],[-0.5,0],[0.05,0]]]
    numbers[9] = [[[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5],[-0.5,0],[0.5,0]]]
    numbers[0] = [[[0,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5],[-0.5,-0.5],[0.05,-0.5]]]
    numbers['dot'] = [[[-0.05,-0.05],[0.05,-0.05],[0.05,0.05],[-0.05,0.05]]]
    
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
    
                numbers_rad[x].append(define_tube((p1,p2),[0,0], rad))
        
        #make each number a single polygon (without holes)
        number_int = MultiPolygon([Polygon([tuple(z) for z in y]) for y in numbers_rad[x]])
        solution = ops.cascaded_union(number_int)
        solution = [np.array(solution.exterior.coords)]
        
        #change the scale of each number
        numbers_rad[x] = [scale*np.array(x) for x in solution]
       
    #create the required number
    full_num = []
    envelope_num = []
    num = str(num)
    len_num = len(num)
    if '.' in num:
        len_num = len_num-1
    
    if len_num==1:
        count = np.array([-scale*1.5,0])
    else:
        count = np.array([-scale*((len_num+1)*1.5)/2,0])
    for x in num:
        if x=='.':
            single_num = []
            for y in numbers_rad['dot']:
                y = np.array([k+count+scale*np.array([0.75,-0.4]) for k in y])
                full_num.append(y)
                single_num.append(y)
        else:
            count = count + np.array([scale*1.5,0])
            single_num = []
            for y in numbers_rad[int(x)]:
                y = np.array([z + count for z in y])
                full_num.append(y)
                single_num.append(y)
    
        #calculate the minimal envelope of each number (not used anymore)
        num_envelope=MultiPolygon([Polygon([tuple(z) for z in y]) for y in single_num])
        envelope = np.array(num_envelope.envelope.exterior.coords)
        envelope_num.append(envelope)
    
    num_result = {}
    num_result['numbers'] = full_num
    num_result['envelopes'] = envelope_num
    
    #Rotate the number by alpha
    if rotation !=0:
        
        alpha = rotation
        R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
    
        for i in range(len(num_result['numbers'])):
            num_result['numbers'][i] = np.squeeze([np.dot([x],R) 
                                                   for x in num_result['numbers'][i]])
        for i in range(len(num_result['envelopes'])):
            num_result['envelopes'][i] = np.squeeze([np.dot([x],R) 
                                                   for x in num_result['envelopes'][i]])
            
            
    #move the number to position pos
    for x in range(len(num_result['numbers'])):
        num_result['numbers'][x] = np.array([z+ np.array(pos) for z in num_result['numbers'][x]])
    for x in range(len(num_result['envelopes'])):
        num_result['envelopes'][x] = np.array([z+ np.array(pos) for z in num_result['envelopes'][x]])
    
    
    return num_result['numbers']


def has_hole(feature):
    """
    Detects holes in features

    Takes a feature (2D numpy array or list of 2D numpy array) and returns the number of holes of the feature

    Parameters
    ----------
    feature : feature
        feature
    
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
 

def reverse_feature(feature, back_square):
    """
    Reverses a number array from polygons to hole within a rectangle.

    Transforms each polygon of a number array into a hole within a given rectangle. Of course the numbers and 
    the rectangle have to be overlapping for holes to be created.

    Parameters
    ----------
    feature : feature
        feature, can contain any element or mix of elements
    back_square : 2D numpy array
        Coordinates of the rectangle in which the features should appear as holes
    
    Returns
    -------
    feature
        original rectangle with holes

    """
    feature1 = MultiPolygon([Polygon([tuple(z) for z in y]) for y in feature])

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
    return back
        
        
        