# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:42:47 2023

@author: helioum

access, load, save, modify, and stack data
"""
import os
import nrrd
import math
import cv2
import shutil
import struct
import numpy as np
import SimpleITK as sitk
from skimage import measure
#%%

def load_images(folder_path, num_img_range, stack=False, grayscale=False, crop_size=[0,0], crop_location = [0,0]):
    ''' Given the files path, it loads them all into one numpy array
    num_images : choose how many images to load (the number of images on the z axis)
    stack : if True, it returns all the images as multi-dimension np array - else, returns a list of images
    crop_size : if not zero, it crops all the images into one same size, in both x and y axes
    grayscale : if True, it loads all the images as grayscale. Else, loads as 3-channel RGB
    create a list of all image names
    '''
    images_list = [f for f in os.listdir(folder_path) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.png')]
    if num_img_range == 'all':
        img_range = [0, len(images_list)]
    elif isinstance(num_img_range, int):
        img_range = [0, num_img_range]
    else:
        img_range = num_img_range
        
    images = []
    for i in range(img_range[0], img_range[1], 1):
        # load the image with desired type (grayscale or RGB)
        img = cv2.imread(os.path.join(folder_path, images_list[i]), (cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR))
    
        if crop_size != [0, 0]:
            x, y = crop_location
            img = img[x:x+crop_size[0], y:y+crop_size[1]]
            
        images.append(img)

    if stack:
        return np.stack(images, axis=0)
    else:
        return images

# given the volume, saves all the slices as images in a created folder
def save_slices(volume, folder_path, number):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
        
    if number == 'all':
        number = len(volume)
    for i in range(number):
        cv2.imwrite(os.path.join(folder_path, 'img' + str(i) + '.png'), volume[i])
        if i > number:
            break

# convert a binary numpy file (centerlines) to obj file
# NOT COMPLETED
# TODO: Add the bifurcation exception
def binary_to_obj(binary, filename):
    verts = np.argwhere(binary)
    line = []
    all_lines = []
    idx = [-2, -2, -2]

    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            for k in range(binary.shape[2]):
                if binary[i, j, k] == 1:
                    if(abs(i - idx[0]) == 1):
                        line.append((i, j, k))
                    elif(abs(j - idx[1]) == 1):
                        line.append((i, j, k))
                    elif(abs(k - idx[2])== 1):
                        line.append((i, j, k))
                    else:
                        line = []
                    if(len(line) != 0):
                        all_lines.append(line)
    
    return all_lines

# loads nrrd file (segmentations from Slicer 3D), and converts it to numpy file
def nrrd2npy(nrrd_path):
    file = nrrd.read(nrrd_path)[0]
    return file

# convertin numpy array to nrrd file
def npy2nrrd(arr, filename):
    nrrd.write(filename, arr)
    
#%%
def resample(data, spacing, output_spacing):    
    image = sitk.GetImageFromArray(data)
    image = sitk.Cast(image, sitk.sitkFloat32)
    image.SetSpacing(spacing)
    size = image.GetSize()
    
    size_scaling = np.array(spacing) / np.array(output_spacing)
    output_size = tuple(int(s * sc) for s, sc in zip(size, size_scaling))
    spacing_scaling = np.array(output_spacing) / np.array(spacing)
    output_origin = tuple(sp * (s - 1) / 2 for s, sp in zip(spacing_scaling, spacing))
    
    resampled = sitk.Resample(
        image,
        output_size,
        outputOrigin=output_origin,
        outputSpacing=output_spacing,
    )
    
    return sitk.GetArrayFromImage(resampled)

def resample_volume(src, new_spacing = [1, 1, 1]):
    interpolator=sitk.sitkLinear
    volume = sitk.GetImageFromArray(src)
    volume = sitk.Cast(volume, sitk.sitkFloat32)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    sitk_volume = sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())
    
    resampled_volume = sitk.GetArrayFromImage(sitk_volume).astype(dtype=np.uint8)
    return resampled_volume

#%%
# convert numpy to OBJ file
def npy2obj(input_file, output_name):
    if isinstance(input_file, str):
        volume = np.load(input_file)
    elif isinstance(input_file, np.ndarray):
        volume = input_file
        input_file = 'skeleton.sth'
    # marching cubes
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.0)
    
    # output_file = input_file.split('.')[0]+'.obj'
    with open(output_name, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

# convert SDP files (created by skelet_kerautret function) to one OBJ file for visualization 
def sdp2obj(input_file):
    with open(input_file+'Vertex.sdp', 'r') as inputfile:
        vlines = inputfile.readlines()
        
    # read edges file
    with open(input_file+'Edges.sdp', 'r') as inputfile:
        enum = inputfile.readlines()
        
    # write in an obj file
    with open(input_file+'.obj', 'w') as outputfile:
        outputfile.write('# vertices\n')
        for line in vlines:
            outputfile.write('v '+ line.strip() + '\n')
        
        outputfile.write('\n# edges\n')
        for edges in enum:
            num1, num2 = edges.strip().split(' ')
            num1 = int(num1)
            num2 = int(num2)
            outputfile.write(f'l {num1 +1} {num2 + 1}\n')

# convert CG file to OBJ
def cg2obj(input_file):
    with open(input_file, 'r') as inputfile: 
        content = inputfile.read()
        
    modified = content.replace('e', 'l')
    with open(input_file.split('.')[0]+'.obj', 'w') as file:
        file.write(modified)
#%%
class vertex:
    def __init__(self, x, y, z, e_out, e_in):
        self.p = np.array([x, y, z])
        self.Eout = e_out
        self.Ein = e_in
class edge:
    def __init__(self, v0, v1, p):
        self.v = (v0, v1)
        self.p = p
class linesegment:
    def __init__(self, p0, p1):
        self.p = (p0, p1)

    #return a set of points sampling the line segment at the specified spacing
    def pointcloud(self, spacing):
        v = self.p[1] - self.p[0]
        l = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        n = int(np.ceil(l / spacing))
        if n == 0:
            return [self.p[0], self.p[1]]
        
        pc = []
        for i in range(n+1):
            pc.append(self.p[0] + v * (i/n))
        return pc
class NWT:
    def __init__(self, filename):        
        [_, fext] = os.path.splitext(filename)                          #get the file extension so that we know the file type
        if fext == ".nwt":                                              #if the file extension is NWT
            self.load_nwt(filename)                                     #load a NWT file
        elif fext == ".obj":                                            #if the file extension is OBJ
            self.load_obj(filename)                                     #load an OBJ file
        else:                                                           #otherwise raise an exception
            raise ValueError("file type is unsupported as a network")
    
    def load_nwt(self, filename):        
        fid = open(filename, "rb")                                      #open a binary file for reading
        self.header = fid.read(14).decode("utf-8")                      #load the header
        self.desc = fid.read(58).decode("utf-8")                        #load the description
        nv = struct.unpack("I", fid.read(4))[0]                         #load the number of vertices and edges
        ne = struct.unpack("I", fid.read(4))[0]

        self.v = []                                                     #create an empty list to store the vertices        
        for _ in range(nv):                                             #iterate across all vertices
            p = np.fromfile(fid, np.float32, 3)                         #read the vertex position
            E = np.fromfile(fid, np.uint32, 2)                          #read the number of edges
            e_out = np.fromfile(fid, np.uint32, E[0])                   #read the indices of the outgoing edges
            e_in = np.fromfile(fid, np.uint32, E[1])                    #read the indices of the incoming edges
            v = vertex(p[0], p[1], p[2], e_out, e_in)                   #create a vertex
            self.v.append(v)

        self.e = []                                                     #create an empty array to store the edges        
        for _ in range(ne):                                             #iterate over all edges            
            v = np.fromfile(fid, np.uint32, 2)                          #load the vertex indices that this edge connects            
            npts = struct.unpack("I", fid.read(4))[0]                   #read the number of points defining this edge            
            pv = np.fromfile(fid, np.float32, 4*npts)                   #read the array of points            
            p = [(pv[i],pv[i+1]) for i in range(0,npts,2)]              #conver the point values to an array of 4-element tuples            
            self.e.append(edge(v[0], v[1], p))                          #create and append the edge to the edge list
    
    def load_obj(self, filename):        
        fid = open(filename, "r")                                       #open the file for reading        
        vertices = []                                                   #create an array of vertices
        lines = []                                                      #create an array of lines
        for line in fid:                                                #for each line in the file
            elements = line.split(" ")                                  #split it into token elements          
            if elements[-1] == '\n' and len(elements) != 1:             #make sure the last element is not \n
                elements.pop(-1)
            if elements[0] == "v":                                      #if the element is a vertex               
                c = [float(i) for i in elements[1:]]                    #get the point coordinates                
                vertices.append(c)                                      #add the coordinates to the vertex list            
            if elements[0] == "l":                                      #if the element is a line             
                idx = [int(i) for i in elements[1:]]                    #get the indices for the points that make up the line                
                lines.append(idx)                                       #add this line to the line list

        self.header = "nwtfileformat "                                  #assign a header and description
        self.desc = "File generated from OBJ"
                                                                        #insert the first and last vertex ID for each line into a set
        vertex_set = set()                                              #create an empty set
        for line in lines:                                              #for each line in the list of lines
            vertex_set.add(line[0])                                     #add the first and last vertex to the vertex set (this will remove redundancies)
            vertex_set.add(line[-1])
        
        obj2nwt = dict()                                                #create a new dictionary - will be used to map vertex IDs in the OBJ to IDs in the NWT object

        #create a mapping between OBJ vertex indices and NWT vertex indices
        vi = 0                                                          #initialize a vertex counter to zero
        for si in vertex_set:                                           #for each vertex in the set of vertices
            obj2nwt[si] = vi                                            #assign the mapping
            vi = vi + 1                                                 #increment the vertex counter

        #iterate through each line (edge), assigning them to their starting and ending vertices
        v_out = [list() for _ in range(len(vertex_set))]                #create an array of empty lists storing the inlet and outlet edges for each vertex
        v_in = [list() for _ in range(len(vertex_set))]

        self.e = []                                                     #create an empty list storing the NWT vertex IDs for each edge (inlet and outlet)
        for li in range(len(lines)):                                    #for each line
            v0 = obj2nwt[lines[li][0]]                                  #get the NWT index for the starting and ending points (vertices)
            v1 = obj2nwt[lines[li][-1]]

            v_out[v0].append(li)                                        #add the line index to a list of inlet edges
            v_in[v1].append(li)                                         #add the line index to a list of outlet edges

            p = []                                                      #create an emptu array of points used to store point positions in the NWT graph
            for pi in range(1, len(lines[li]) - 1):                     #for each point in the line that is not an end point (vertex)
                p.append(np.array(vertices[lines[li][pi]-1]))              #add the coordinates for that point as a tuple into the point list
            self.e.append(edge(v0, v1, p))                              #create an edge, specifying the inlet and outlet vertices and all defining points

        #for each vertex in the set, create a NWT vertex containing all of the necessary edge information
        self.v = []                                                     #create an empty list to store the vertices
        for s in vertex_set:                                            #for each OBJ vertex in the vertex set
            vi = obj2nwt[s]                                             #calculate the corresponding NWT index
            self.v.append(vertex(vertices[s-1][0], vertices[s-1][1], vertices[s-1][2], v_out[vi], v_in[vi]))    #create a vertex object, consisting of a position and attached edges

    #return a set of line segments connecting all points in the network
    def linesegments(self):

        s = []                                                          #create an empty list of line segments
        for e in self.e:                                                #for each edge in the graph
            p0 = self.v[e.v[0]].p                                       #load the first point (from the starting vertex)

            for p in e.p:                                               #for each point in the edge
                p1 = np.array([p[0], p[1], p[2]])                       #get the second point for the line segment
                s.append(linesegment(p0, p1))                           #append the line segment to the list of line segments
                p0 = p1                                                 #update the start point for the next segment to the end point of this one
            
            p1 = self.v[e.v[1]].p                                       #load the last point (from the ending vertex)
            s.append(linesegment(p0, p1))                               #append the last line segment for this edge to the list
        return s

    #return a point cloud sampling the centerline of the network at the given spacing
    def pointcloud(self, spacing):
        ls = self.linesegments()
        print(len(ls))
        pc = []
        for l in ls:
            pc = pc + l.pointcloud(spacing)
        return pc
    
    def save_obj(self, output_name):
        with open(output_name, 'w') as output:
            output.write('# vertices\n')
            for vert in self.v:             #vertex variable
                x, y, z = vert.p
                output.write(f'v {x} {y} {z}\n')
            output.write('# edges\n')
            for edg in self.e:
                v1, v2 = edg.v
                output.write(f'l {v1+1} {v2+1}\n')


#%%
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)