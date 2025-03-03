import numpy as np
from numpy.ma import shape
from scipy.sparse import data
import mesh
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from glob import glob


def hausdorff_distance(mesh1:mesh.Mesh, mesh2:mesh.Mesh) -> float:

    """
    Computes the Hausdorff distance between two point sets using NumPy.

    Parameters:
        mesh1 (Mesh): mesh class defined from a .vtk file 
        mesh2 (Mesh): mesh class defined from a .vtk file  

    Returns:
        float: The Hausdorff distance.
    """

    pointsA = mesh1.points
    pointsB = mesh2.points
    
    # For each point in pointsA, find the distance to the closest point in pointsB.
    diff_A_to_B = pointsA[:, np.newaxis, :] - pointsB[np.newaxis, :, :]
    dists_A_to_B = np.linalg.norm(diff_A_to_B, axis=2)
    directed_A_to_B = np.max(np.min(dists_A_to_B, axis=1))
    
    # For each point in pointsB, find the distance to the closest point in pointsA.
    diff_B_to_A = pointsB[:, np.newaxis, :] - pointsA[np.newaxis, :, :]
    dists_B_to_A = np.linalg.norm(diff_B_to_A, axis=2)
    directed_B_to_A = np.max(np.min(dists_B_to_A, axis=1))
    
    # The Hausdorff distance is the maximum of the two directed distances.
    return max(directed_A_to_B, directed_B_to_A)

def compare_shape_dna(mesh1:mesh.Mesh, mesh2:mesh.Mesh) -> float:
    distance = euclidean(mesh1.eigenvalues, mesh2.eigenvalues)
    return distance

shapes = ["sphere", "deformed_sphere", "shifted_sphere", "pyramid", "inverted_pyramid", "cube", "shifted_cube"]

path = "data/simple_shapes"

datasets = [mesh.Mesh(f"{path}/{shape}.vtk") for shape in shapes]

n_datasets = len(shapes)

hausdorff_distance_matrix = np.zeros((n_datasets, n_datasets))
shapedna_distance_matrx = np.zeros((n_datasets, n_datasets))

for i in range(n_datasets):
    for j in range(n_datasets):
        hausdorff_distance_matrix[i,j] = hausdorff_distance(datasets[i], datasets[j])
        shapedna_distance_matrx[i,j] = compare_shape_dna(datasets[i], datasets[j])

hausdorff_dist_df = pd.DataFrame(hausdorff_distance_matrix,index=shapes,columns=shapes)
shapedna_dist_df = pd.DataFrame(shapedna_distance_matrx, index=shapes,columns=shapes)

fig, ax = plt.subplots(1,2, figsize=(12,6))
sns.heatmap(hausdorff_dist_df, ax=ax[0], annot=True )
ax[0].set_title("Hausdorff distance")

sns.heatmap(shapedna_dist_df, ax=ax[1],annot=True, fmt=".1f")
ax[1].set_title("ShapeDNA")

plt.tight_layout()
plt.savefig("figs/results/distance_matrices.png", dpi=300)
#plt.show()
plt.close()

###dimensionality reduction

deformed_spheres_paths = glob("data/simple_shapes/*deformed_sphere*")
deformed_spheres = [mesh.Mesh(path) for path in deformed_spheres_paths]

spheres_paths = list(set(glob("data/simple_shapes/*sphere*")) - set(glob("data/simple_shapes/*deformed_sphere*")))
spheres = [mesh.Mesh(path) for path in spheres_paths]

deformed_cubes_paths = glob("data/simple_shapes/*deformed_cube*")
deformed_cubes = [mesh.Mesh(path) for path in deformed_cubes_paths]

cubes_paths = list(set(glob("data/simple_shapes/*cube*")) - set(glob("data/simple_shapes/*deformed_cube*")))
cubes = [mesh.Mesh(path) for path in cubes_paths]

deformed_pyramids_paths = glob("data/simple_shapes/*deformed_pyramid*")
deformed_pyramids = [mesh.Mesh(path) for path in deformed_pyramids_paths]
print(deformed_pyramids_paths)

def_sph_eigs = np.array([o.eigenvalues for o in deformed_spheres]).T
sph_eigs = np.array([o.eigenvalues for o in spheres]).T
def_c_eigs = np.array([o.eigenvalues for o in deformed_cubes]).T
c_eigs = np.array([o.eigenvalues for o in cubes]).T
def_p_eigs = np.array([o.eigenvalues for o in deformed_pyramids]).T

print(def_sph_eigs.shape, sph_eigs.shape, def_c_eigs.shape, c_eigs.shape, def_p_eigs.shape)

all_data = np.hstack([def_sph_eigs, sph_eigs, def_c_eigs, c_eigs, def_p_eigs])
print(all_data.shape)

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5, early_exaggeration=5).fit_transform(all_data )


categories = {
    "deformed_sphere": (0, 11, "blue"),
    "sphere": (11, 23, "red"),
    "deformed_cubes": (23, 33, "green"),
    "cubes": (33, 45, "purple"),
    "deformed_pyramids": (45, 55, "orange"),
}

plt.figure(figsize=(8, 6))


for label, (start, end, color) in categories.items():
    print(label, start, end)
    plt.scatter(X_embedded[start:end, 0], X_embedded[start:end, 1], 
                label=label, color=color, alpha=0.7, edgecolors="k", s=46)

# Formatting
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Eigenvalues embedding")
plt.legend()
#plt.grid(True)
plt.savefig("figs/results/tsne_embedding.png", dpi=300)
plt.show()
plt.close()
