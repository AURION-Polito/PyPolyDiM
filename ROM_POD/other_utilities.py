import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.tri

def make_np_sparse(A_sparse_data, new_size = None, shifts = None, transpose = None):
    if new_size is None:
        new_size = [A_sparse_data.size[0], A_sparse_data.size[1]]
    if shifts is None:
        shifts = [0, 0]
    if transpose is None:
        transpose = False
    if transpose:
        return scipy.sparse.csc_array((A_sparse_data.values, 
                                       ([i + shifts[1] for i in A_sparse_data.cols],
                                        [i + shifts[0] for i in A_sparse_data.rows])), 
                                      shape=(new_size[0], new_size[1]))
    else:
        return scipy.sparse.csc_array((A_sparse_data.values, 
                                       ([i + shifts[0] for i in A_sparse_data.rows],
                                        [i + shifts[1] for i in A_sparse_data.cols])), 
                                      shape=(new_size[0], new_size[1]))
    
                                  
def plot_mesh(mesh, export_folder = ""):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_aspect('equal')
    
    coordinates = mesh.cell0_ds_coordinates()
    ax1.triplot(matplotlib.tri.Triangulation(coordinates[0, :], coordinates[1, :]), 'ko-', lw=1)
    ax1.grid(True)

    if export_folder != "":
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        file_name = 'Mesh.png'
        file_path = os.path.join(export_folder, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close(fig)
    else:
        plt.pause(0.1)
        plt.close(fig)

def evaluate_function_on_points(points, function_name):
    num_points = points.shape[1]
    function_values = np.zeros(num_points)

    for p in range(1, num_points):        
      function_values[p] = function_name(points[0, p], points[1, p], points[2, p])
    return function_values

def plot_solution(mesh, solution_cell0Ds, title = None, export_folder = None):
    plot_solution_on_coordinates(mesh.cell0_ds_coordinates(), solution_cell0Ds, title, export_folder)

def plot_solution_on_coordinates(coordinates, solution_on_coordinates, title = None, export_folder = None):
    if title is None:
        title = "Solution"
    if export_folder is None:
        export_folder = ""
    
    x = coordinates[0,:]
    y = coordinates[1,:]
    z = solution_on_coordinates
    triang = matplotlib.tri.Triangulation(x, y)
    
    fig = plt.figure(figsize = plt.figaspect(0.5))
    fig.suptitle(title)
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(triang, z, shading='flat')
    ax1.triplot(matplotlib.tri.Triangulation(coordinates[0, :], coordinates[1, :]), 'k--', lw=1)
    fig.colorbar(tpc)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_trisurf(x, y, z, triangles=triang.triangles, cmap=plt.cm.Spectral)
    
    if export_folder != "": 
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        file_name = title + '.png'
        file_path = os.path.join(export_folder, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close(fig)
    else:
        plt.pause(0.1)
        plt.close(fig)
