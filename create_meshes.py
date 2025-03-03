#! ~/coding/keyward/bin/python
"""Generates some simple trimeshes, saves them as .vtk, plots them and saves figures"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from typing import Tuple
import os.path
from glob import glob
import os.path

def create_sphere(centre:Tuple[float,float,float]=(0,0,0)):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(30)
    sphere.SetPhiResolution(30)
    sphere.SetCenter(*centre)
    
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputConnection(sphere.GetOutputPort())
    triangulator.Update()
    
    return triangulator.GetOutput()

def create_deformed_sphere(centre=(0,0,0), noise_scale=0.073, resolution=30, seed=42):
    """
    Create a deformed sphere with triangular mesh
    Parameters:
        noise_scale: Deformation magnitude (0.1-0.5 recommended)
        resolution: Angular resolution of base sphere
        seed: Random seed for reproducible deformation
    """

    # Add noise using point displacement
    polydata = create_sphere(centre=centre)
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    
    # Generate structured noise for organic deformation
    #np.random.seed(seed)
    angles = np.arctan2(points[:,1], points[:,0])  # Azimuthal angle
    zenith = np.arccos(points[:,2]/np.linalg.norm(points, axis=1))  # Polar angle
    
    # Create noise pattern using spherical coordinates
    noise = (
        np.sin(8 * angles) * np.cos(4 * zenith) + 
        0.5 * np.sin(16 * angles) * np.cos(8 * zenith)
    ) * noise_scale

    # Displace points along radial direction
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    deformed_points = points * (1 + noise[:, np.newaxis]/norms)

    # Update points in VTK
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(deformed_points, deep=True))
    polydata.SetPoints(vtk_points)

    # Clean and recalculate normals
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaner.GetOutputPort())
    normals.ComputeCellNormalsOn()
    normals.Update()

    return normals.GetOutput()

def create_cube(divisions=5, xmin:float=-1,xmax:float=1, ymin:float=-1, ymax:float=1, zmin:float=-1,zmax:float=1):
    """Create cube with adjustable resolution using tessellation"""
    box = vtk.vtkTessellatedBoxSource()
    box.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)  
    box.SetLevel(divisions)  
    box.QuadsOff()  # Force triangular faces
    box.Update()
    
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(box.GetOutputPort())
    
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputConnection(cleaner.GetOutputPort())
    triangulator.Update()
    
    return triangulator.GetOutput()

def create_deformed_cube(divisions=5, noise_scale=0.2, xmin:float=-1,xmax:float=1, ymin:float=-1, ymax:float=1, zmin:float=-1,zmax:float=1):
    """Create cube with wavy surface deformation"""
    # Generate base cube
    cube = create_cube(divisions, xmin,xmax, ymin, ymax, zmin,zmax)
    points = vtk_to_numpy(cube.GetPoints().GetData())
    
    # Generate 3D noise pattern
    #np.random.seed(seed)
    x, y, z = points.T
    noise = (
        np.sin(4*x + np.random.uniform(0, 2*np.pi)) * 
        np.cos(4*y + np.random.uniform(0, 2*np.pi)) * 
        np.sin(4*z + np.random.uniform(0, 2*np.pi))
    ) * noise_scale

    # Displace points along surface normals
    deformed_points = points + noise[:, np.newaxis] * points/np.linalg.norm(points, axis=1, keepdims=True)

    # Update geometry
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(deformed_points, deep=True))
    cube.GetPoints().SetData(vtk_points.GetData())

    # Process normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(cube)
    normals.ComputeCellNormalsOn()
    normals.Update()
    
    return normals.GetOutput()

def create_pyramid(subdivisions=3, inverted=False):
    """Create pyramid with swapable base/apex positions"""
    # Configure base and apex heights
    base_z = 2.0 if inverted else 0.0
    apex_z = 0.0 if inverted else 2.0

    # Create points with configurable z positions
    points = vtk.vtkPoints()
    points.InsertNextPoint(-1, -1, base_z)  # Base corner 0
    points.InsertNextPoint(1, -1, base_z)   # Base corner 1
    points.InsertNextPoint(1, 1, base_z)    # Base corner 2
    points.InsertNextPoint(-1, 1, base_z)   # Base corner 3
    points.InsertNextPoint(0, 0, apex_z)    # Apex

    cells = vtk.vtkCellArray()
    # Base triangles (wind clockwise for consistent normals)
    cells.InsertNextCell(3, [0, 3, 2] if inverted else [0, 1, 2])
    cells.InsertNextCell(3, [0, 2, 1] if inverted else [0, 2, 3])
    
    # Side triangles (maintain outward-facing normals)
    side_connections = [
        [0, 3, 4] if inverted else [0, 1, 4],
        [3, 2, 4] if inverted else [1, 2, 4],
        [2, 1, 4] if inverted else [2, 3, 4],
        [1, 0, 4] if inverted else [3, 0, 4]
    ]
    for connection in side_connections:
        cells.InsertNextCell(3, connection)

    pyramid = vtk.vtkPolyData()
    pyramid.SetPoints(points)
    pyramid.SetPolys(cells)

    # Processing pipeline
    subdiv = vtk.vtkLinearSubdivisionFilter()
    subdiv.SetInputData(pyramid)
    subdiv.SetNumberOfSubdivisions(subdivisions)
    
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputConnection(subdiv.GetOutputPort())
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(triangulator.GetOutputPort())
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()

def create_deformed_pyramid(subdivisions=3, twist_angle=30, noise_scale=0.3, seed=42):
    """Create pyramid with rotational twist and surface noise"""
    # Generate base pyramid
    pyramid = create_pyramid(subdivisions)
    points = vtk_to_numpy(pyramid.GetPoints().GetData())
    
    # Add rotational twist
    #np.random.seed(seed)
    theta = np.arctan2(points[:,1], points[:,0])
    z = points[:,2]
    twist = twist_angle * np.pi/180 * z  # Convert degrees to radians
    
    # Apply twist rotation
    rot_x = points[:,0] * np.cos(twist) - points[:,1] * np.sin(twist)
    rot_y = points[:,0] * np.sin(twist) + points[:,1] * np.cos(twist)
    
    # Add surface noise
    noise = noise_scale * np.random.randn(len(points))
    deformed_points = np.column_stack([
        rot_x + noise,
        rot_y + noise,
        z + 0.5*noise  # Add vertical deformation
    ])

    # Update geometry
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(deformed_points, deep=True))
    pyramid.GetPoints().SetData(vtk_points.GetData())

    # Process and clean
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pyramid)
    
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaner.GetOutputPort())
    normals.ComputeCellNormalsOn()
    normals.Update()
    
    return normals.GetOutput()



def save_vtk(polydata, filename):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.SetFileTypeToASCII()
    writer.Write()

def read_vtk(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    points_array = vtk_to_numpy(points.GetData()) if points else np.array([])
    
    faces_array = []
    if polydata.GetPolys().GetNumberOfCells() > 0:
        cells = polydata.GetPolys()
        cells.InitTraversal()
        id_list = vtk.vtkIdList()
        while cells.GetNextCell(id_list):
            if id_list.GetNumberOfIds() == 3:  # Only triangles
                faces_array.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
    
    return points_array, np.array(faces_array)



def plot_point_cloud(filename, figsize=(8,6)):
    
    points, _ = read_vtk(filename)
    fig = plt.figure(figsize = figsize) 
     
    ax = fig.add_subplot( projection='3d')
    ax.scatter3D(points[:,0], points[:,1] , points[:,2], color = "green")
    name = os.path.basename(filename).split('.')[0]
    # Set axis limits and labels
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    plt.title(f"{name} point cloud")
    plt.tight_layout() 
    # show plot
    plt.savefig(f"figs/point_cloud_{name}.png", dpi=300)
    #plt.show()
    plt.close()


def plot_mesh_matplotlib(filename, figsize=(8, 6)):
    # Read VTK file
    points, faces = read_vtk(filename)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create collection of triangular faces
    triangles = points[faces]
    mesh = Poly3DCollection(triangles, alpha=0.8, edgecolor='k', linewidths=0.5)
    mesh.set_facecolor('cyan')
    
    # Add mesh to plot
    ax.add_collection3d(mesh)
    
    # Set axis limits and labels
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    name = os.path.basename(filename).split('.')[0]
    plt.title(f"{name} mesh")
    plt.savefig(f"figs/{name}.png", dpi=300)
    #plt.show()
    plt.close()

if __name__ == "__main__":
    
    shapes = {'sphere': create_sphere(),
              'shifted_sphere': create_sphere(centre=(0,0,1)),
              'deformed_sphere': create_deformed_sphere(),
              'cube': create_cube(),
              'shifted_cube': create_cube(xmin=0.5,xmax=2), 
              'pyramid': create_pyramid(inverted=False),
              'inverted_pyramid': create_pyramid(inverted=True),}

    for name, shape in shapes.items():
        save_vtk(shape, f"data/simple_shapes/{name}.vtk")

    for name in shapes.keys():
        plot_point_cloud(f"data/simple_shapes/{name}.vtk")
        plot_mesh_matplotlib(f"data/simple_shapes/{name}.vtk")
        pass

    ##generate some more data
    for i in range(10):
        x,y,z = np.random.uniform(-3,3,size=3)
        _sphere = create_sphere((x,y,z))
        save_vtk(_sphere, f"data/simple_shapes/random_sphere_{i}.vtk")

    for i in range(10):    
        x,y,z = np.random.uniform(-3,3,size=3)
        def_level = np.random.uniform(0.05, 0.13, 1)
        _sphere = create_deformed_sphere((x,y,z), noise_scale=def_level)
        save_vtk(_sphere, f"data/simple_shapes/random_deformed_sphere_{i}.vtk") 

    for i in range(10):
        x0,x1, y0,y1, z0,z1 = np.random.uniform(-3,3,size=6)
        _cube = create_cube(divisions=5,xmin=x0,xmax=x1,ymin=y0,ymax=y1,zmin=z0,zmax=z1)
        save_vtk(_cube, f"data/simple_shapes/random_cube_{i}.vtk")
        
    for i in range(10):
        x0,x1, y0,y1, z0,z1 = np.random.uniform(-3,3,size=6)
        def_level = np.random.uniform(0.05, 0.13, 1)
        _cube = create_deformed_cube(divisions=5, noise_scale=def_level,xmin=x0,xmax=x1,ymin=y0,ymax=y1,zmin=z0,zmax=z1)
        save_vtk(_cube, f"data/simple_shapes/random_deformed_cube_{i}.vtk")

          
    for i in range(10):
        twist_level = np.random.uniform(40,50, 1) 
        def_level = np.random.uniform(0.05, 0.13, 1)
        _pyr = create_deformed_pyramid(twist_angle=twist_level, noise_scale=def_level) 
        save_vtk(_pyr, f"data/simple_shapes/random_deformed_pyramid_{i}.vtk")
    
    all_random_shapes = glob("data/simple_shapes/random_*")
    for name in all_random_shapes:
        _save_name = os.path.basename(name).split(".")[0]

        plot_point_cloud(f"data/simple_shapes/{_save_name}.vtk")
        plot_mesh_matplotlib(f"data/simple_shapes/{_save_name}.vtk")

