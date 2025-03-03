from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from scipy.sparse import csc_matrix
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from typing import Tuple
import sys



@dataclass
class Mesh():
    filename:str
    reader:vtk.vtkPolyDataReader = field(init=False)
    points:np.ndarray = field(init=False)
    faces:np.ndarray = field(init=False)
    num_points:int = field(init=False)
    eigenvalues:np.ndarray = field(init=False)
    eigenvectors:np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.reader = self._open_vtk()


        points, faces,  num_p = self._get_mesh_data()
        self.points = points
        self.faces = faces
        self.num_points = num_p

        evals, evecs = self.laplace_beltrami_eigenvalue()
        self.eigenvalues = evals
        self.eigenvectors = evecs

    def _open_vtk(self):
        """
        Reads a VTK file and returns a vtkPolyData object.
        """
        filename = self.filename
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        return reader

    def _get_mesh_data(self) -> Tuple[np.ndarray, np.ndarray, int]:

        polydata = self.reader.GetOutput()
        # Extract points
        points = polydata.GetPoints()
        points_array = vtk_to_numpy(points.GetData()) if points else np.array([])
                # Extract faces (triangles)
        faces_array = []
        if polydata.GetPolys().GetNumberOfCells() > 0:
            cells = polydata.GetPolys()
            cells.InitTraversal()
            id_list = vtk.vtkIdList()
            while cells.GetNextCell(id_list):
                if id_list.GetNumberOfIds() == 3:  # Only triangles
                    faces_array.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
        
        num_points = polydata.GetNumberOfPoints()

        return points_array, np.array(faces_array), num_points


    def laplace_beltrami_eigenvalue(self,  k:int=50) -> Tuple[np.ndarray, np.ndarray]:

        """Inspired by https://github.com/Deep-MI/LaPy/blob/5d0cc267125f503cb4ba3f44a03fb076dc8f3507/lapy/solver.py"""

        points_array = self.points
        faces_array = self.faces

        
        face_idx_list = [faces_array[:,i] for i in range(3)]
        pts_coord_list = [points_array[idx,:] for idx in face_idx_list]

        diff_vector_list = [pts_coord_list[1] - pts_coord_list[0], pts_coord_list[2] - pts_coord_list[1], pts_coord_list[0] - pts_coord_list[2]]

        cross_product = np.cross(diff_vector_list[1], diff_vector_list[2])


        vol = 2 * np.sqrt(np.sum(cross_product * cross_product, axis=1))
        vol_mean =  1e-4* np.mean(vol)
        vol[vol < sys.float_info.epsilon] = vol_mean

        #off diagonals:
        a12 = np.sum(diff_vector_list[1] * diff_vector_list[2], axis=1) / vol
        a23 = np.sum(diff_vector_list[2] * diff_vector_list[0], axis=1) / vol
        a31 = np.sum(diff_vector_list[0] * diff_vector_list[1], axis=1) / vol

        #diagonals
        a11 = -a12 - a31
        a22 = -a12 - a23
        a33 = -a31 - a23

        local_a = np.column_stack([a12, a12, a23, a23, a31, a31, a11, a22, a33]).reshape(-1)

        i = np.column_stack([face_idx_list[0], face_idx_list[1], face_idx_list[1], face_idx_list[2], face_idx_list[2], face_idx_list[0], face_idx_list[0], face_idx_list[1], face_idx_list[2]]).reshape(-1)
        j = np.column_stack([face_idx_list[1], face_idx_list[0], face_idx_list[2], face_idx_list[1], face_idx_list[0], face_idx_list[2], face_idx_list[0], face_idx_list[1], face_idx_list[2]]).reshape(-1)
        
        A = csc_matrix((local_a, (i, j)))

        b_ii = vol / 24
        b_ij = vol / 48
        local_b = np.column_stack(
            (b_ij, b_ij, b_ij, b_ij, b_ij, b_ij, b_ii, b_ii, b_ii)
        ).reshape(-1)
        B  = csc_matrix((local_b, (i, j)))

        
        sigma = -0.01
        lu = splu(A - sigma * B)
        op_inv = LinearOperator(matvec=lu.solve, shape=A.shape, dtype=A.dtype)
        eigenvalues, eigenvectors = eigsh(A, k,B, sigma=sigma, OPinv=op_inv)

        return eigenvalues, eigenvectors


