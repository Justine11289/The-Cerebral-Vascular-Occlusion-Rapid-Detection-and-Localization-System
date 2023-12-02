import argparse
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from stl import mesh
import open3d as o3d
from scipy.spatial import KDTree
from pyntcloud import PyntCloud
from nibabel.testing import data_path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process NIfTI files.')
    parser.add_argument('--input_file', type=str, help='Path to the NIfTI file.')
    return parser.parse_args()

args = parse_arguments()
input_file = args.input_file
file_path = ''
file_dir = ''
no_frame_file = ''
left_file = ''
right_file = ''
left_mirror = ''
right_mirror = ''

"""Erosion"""
def findstartm(mat,res):
    # This part is to find the start coordinate"
    startm = (0,0)
    mt = np.transpose(mat)
    start_found = False
    for i in range(len(mt)):
        for j in range(len(mt[i])):
            if start_found == False:
                if mt[i][j] == 0:
                    start_found = True
                    startm = j,i
    # End of this part is to find the start coordinate"
    res = findPath(mat,res,startm)
    return res

def findPath(mat, res, startm):
    visited = set()
    stack = [(startm, (-1, -1))]
    while stack:
        coordinates, parent = stack.pop()
        if coordinates == (-1, -1) or mat[coordinates[0]][coordinates[1]] == 1:
            continue
        if coordinates in visited:
            continue

        visited.add(coordinates)
        res[coordinates[0]][coordinates[1]] = 0

        # Explore neighbor nodes
        for direction in ['n', 's', 'w', 'e', 'a', 'b', 'c', 'd']:
            neighbor_coordinates = getNeighbourCoordinates(coordinates, direction, mat)
            if neighbor_coordinates != (-1, -1) and neighbor_coordinates not in visited:
                stack.append((neighbor_coordinates, coordinates))

    return res

def getNeighbourCoordinates(coordinates, direction, mat):
    i, j = coordinates

    if direction == 'n':
        return (i - 1, j) if i - 1 >= 0 else (-1, -1)
    elif direction == 'a':
        return (i - 1, j + 1) if i - 1 >= 0 and j + 1 < len(mat[i]) else (-1, -1)
    elif direction == 'e':
        return (i, j + 1) if j + 1 < len(mat[i]) else (-1, -1)
    elif direction == 'b':
        return (i + 1, j + 1) if i + 1 < len(mat) and j + 1 < len(mat[i]) else (-1, -1)
    elif direction == 's':
        return (i + 1, j) if i + 1 < len(mat) else (-1, -1)
    elif direction == 'c':
        return (i + 1, j - 1) if i + 1 < len(mat) and j - 1 >= 0 else (-1, -1)
    elif direction == 'w':
        return (i, j - 1) if j - 1 >= 0 else (-1, -1)
    elif direction == 'd':
        return (i - 1, j - 1) if i - 1 >= 0 and j - 1 >= 0 else (-1, -1)
    else:
        return (-1, -1)

def clean(shape, res, res2):
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            x1 = res2[i-1][j-1]
            x2 = res2[i][j-1]
            x3 = res2[i+1][j-1]
            x4 = res2[i-1][j]
            x5 = res2[i+1][j]
            x6 = res2[i-1][j+1]
            x7 = res2[i][j+1]
            x8 = res2[i+1][j+1]
            total = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
            if total < 8:
                res[i][j] = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i==0 or j==0 or i==shape[0]-1 or j==shape[1]-1:
                res[i][j] = 0
    return res

"""Cut half the brain"""
def half(nifti_file):
    global left_file, right_file
    data = np.loadtxt(nifti_file)
    data = ptsrange(data)
    min_x = data[:, 0].min()
    max_x = data[:, 0].max()
    cut = 5/8
    center_l = int(max_x - (max_x-min_x) *cut)
    center_r = int(min_x + (max_x-min_x) *cut)
    sorted_coordinates = data[data[:, 0].argsort()]
    left_half = []
    right_half = []
    for d in sorted_coordinates:
        if d[0] > center_l: left_half.append(d)
        if d[0] < center_r: right_half.append(d)
    left_file = file_dir + 'left_' + os.path.basename(no_frame_file.split('/')[-1])
    np.savetxt(left_file, left_half, delimiter=' ', fmt='%.6f')
    right_file = file_dir+ 'right_' + os.path.basename(no_frame_file.split('/')[-1])
    np.savetxt(right_file, right_half, delimiter=' ', fmt='%.6f')
    mirror(left_file,right_file,center_l,center_r)

# Limit PTS file range
def ptsrange(pts):
    min_x = pts[:, 0].min()
    max_x = pts[:, 0].max()
    min_y = pts[:, 1].min()
    max_y = pts[:, 1].max()
    min_z = pts[:, 2].min()
    max_z = pts[:, 2].max()
    offset_x = (min_x+max_x)/5
    offset_y = (min_y+max_y)/5
    offset_z = (min_z+max_z)/10
    within_range = (
        (pts[:, 0] >= min_x + offset_x) & (pts[:, 0] <= max_x - offset_x) &
        (pts[:, 1] >= min_y) & (pts[:, 1] <= max_y - offset_y) &
        (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z - offset_z)
    )
    pts_within_range = pts[within_range]
    return pts_within_range

"""Mapping"""
def mirror(left_pts, right_pts, left_shaft, right_shaft):
    global left_mirror, right_mirror
    shaft = right_shaft-left_shaft
    """Left"""
    left_vertices = np.loadtxt(left_pts, dtype=float)
    # Mirroring
    left_mirrored = np.copy(left_vertices)
    left_mirrored[:, 0] = 2*right_shaft - left_mirrored[:, 0] - shaft
    # Save the mirrored PTS file
    left_mirror = file_dir + 'mirror_' + os.path.basename(left_pts.split('/')[-1])
    np.savetxt(left_mirror, left_mirrored, delimiter=' ', fmt='%.6f')

    """Right"""
    right_vertices = np.loadtxt(right_pts, dtype=float)
    # Mirroring
    right_mirrored = np.copy(right_vertices)
    right_mirrored[:, 0] = 2*left_shaft - right_mirrored[:, 0] + shaft
    # Save the mirrored PTS file
    right_mirror = file_dir + 'mirror_' + os.path.basename(right_pts.split('/')[-1])
    np.savetxt(right_mirror, right_mirrored, delimiter=' ', fmt='%.6f')

"""Convert"""
def nifti_to_stl(nifti_file):
    # Get NIfTI data
    data = nifti_file.get_fdata()
    affine_matrix = nifti_file.affine
    # Convert NIfTI data to binary image (0 : air, !=0 : solid)
    binary_image = np.where(data != 0, 1, 0).astype(np.uint8)
    # Use Marching Cubes algorithm to extract STL triangular mesh
    verts, faces, _, _ = measure.marching_cubes(binary_image)
    # Create STL
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = affine_matrix.dot(verts[face[j]].tolist() + [1])[:3]
    stl_file = nifti_file.get_filename().replace(".nii.gz",".stl")
    # Save STL
    stl_mesh.save(stl_file)
    pts_file = stl_to_pts(stl_file)
    return pts_file

def stl_to_pts(stl_file):
    # Get STL data
    mesh = o3d.io.read_triangle_mesh(stl_file)
    # Save PLY
    ply_file = stl_file.replace('.stl', '.ply')
    o3d.io.write_triangle_mesh(ply_file, mesh)
    anky = PyntCloud.from_file(ply_file)
    anky.points
    number_of_points = 8000 # Number of points
    anky_cloud = anky.get_sample("mesh_random", n=number_of_points, rgb=False, normals=True, as_PyntCloud=True)
    # Save PTS
    pts_file = ply_file.replace('.ply', f'_{number_of_points}.pts')
    anky_cloud.to_file(pts_file,sep=" ", header=0, index=0)
    return pts_file

def pts_to_pcd(data):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:,:3])
    return point_cloud

"""Iterative Closest Point"""
def ICP(pts_file, mirror_file):
    min_avg_distance = float('inf')
    best_threshold = None
    best_transformed_brain_pcd = None
    # Get PTS data
    original = np.loadtxt(pts_file)
    mirror = np.loadtxt(mirror_file)
    # Traverse threshold values
    for threshold in range(1, 21):
        target = gravity(original)
        source = gravity(mirror)
        # Get the transformation matrix
        transformation = apply_icp(source, target, threshold)
        # Apply transformation matrix
        original_pcd = pts_to_pcd(mirror)
        transformed_pcd = original_pcd.transform(transformation)
        compare_pcd = pts_to_pcd(original)
        avg_distance = compute_average_distance(transformed_pcd, compare_pcd)
        # Update the threshold of the minimum average distance and the converted point cloud
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            best_threshold = threshold
            best_transformed_pcd = transformed_pcd
    print(best_threshold)
    points = np.asarray(best_transformed_pcd.points)
    np.savetxt(mirror_file, points, delimiter=' ', fmt='%.6f')

# The center of gravity
def gravity(data):
    # Cut into 10x10x10 squares
    x_bins = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 10).astype(float)
    y_bins = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 10).astype(float)
    z_bins = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 10).astype(float)

    # Generate index of small squares
    indices_x = np.digitize(data[:, 0], bins=x_bins)
    indices_y = np.digitize(data[:, 1], bins=y_bins)
    indices_z = np.digitize(data[:, 2], bins=z_bins)

    indices = indices_x + 10 * (indices_y - 1) + 10 * 10 * (indices_z - 1)

    # Calculate the center of gravity of each small square
    centroids = []
    for i in range(1, 11):
        for j in range(1, 11):
            for k in range(1, 11):
                mask = (indices == i + 10 * (j - 1) + 10 * 10 * (k - 1))
                if np.sum(mask) > 0:
                    centroid = np.mean(data[mask], axis=0)
                    centroids.append(centroid)

    centroids = np.array(centroids)
    point_cloud = pts_to_pcd(centroids)
    return point_cloud

# Transformation matrix
def apply_icp(source, target, threshold=5, max_iteration=40):
    # Initialize the transformation matrix to the identity matrix
    init_transformation = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    return reg_p2p.transformation

# Average shortest distance
def compute_average_distance(source_pcd, target_pcd):
    # Find the nearest point using KDTree
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    distances = []

    for point in source_pcd.points:
        [_, idx, _] = tree.search_knn_vector_3d(point, 1)
        closest_point = target_pcd.points[idx[0]]
        distance = np.linalg.norm(np.array(point) - np.array(closest_point))
        distances.append(distance)

    # Calculate average distance
    avg_distance = np.mean(distances)
    return avg_distance

"""KDtree"""
def KDtree_subtraction(pts1, pts2):
    dis = 6
    points1 = np.loadtxt(pts1)
    points2 = np.loadtxt(pts2)
    """Find the shortest distance between each point in the first point set pts1 and the second point set pts2"""
    kdtree = KDTree(points1)
    distances, indices = kdtree.query(points2)
    new_pts_file = file_dir + "find_" + pts1.split("/")[-1]
    new_pts = points2[distances > dis]
    np.savetxt(new_pts_file, new_pts)
    filter_pts(new_pts_file,new_pts_file, cube_size=10, threshold=15)

# Filter
def filter_pts(input_path, output_path, cube_size=10, threshold=15):
    points = []
    # Get PTS data
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Read XYZ coordinates
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append((x, y, z))

    filtered_points = []
    # Find the coordinate range
    min_x = min(points, key=lambda t: t[0])[0]
    min_y = min(points, key=lambda t: t[1])[1]
    min_z = min(points, key=lambda t: t[2])[2]

    max_x = max(points, key=lambda t: t[0])[0]
    max_y = max(points, key=lambda t: t[1])[1]
    max_z = max(points, key=lambda t: t[2])[2]

    # Iterate over each small cubic space
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            z = min_z
            while z < max_z:
                # Find all points in small cubic space
                cube_points = [
                    point for point in points
                    if x <= point[0] < x + cube_size and
                    y <= point[1] < y + cube_size and
                    z <= point[2] < z + cube_size
                ]
                if len(cube_points) >= threshold:
                    filtered_points.extend(cube_points)

                z += cube_size
            y += cube_size
        x += cube_size

    # Save as new PTS file
    with open(output_path, 'w') as f:
        for point in filtered_points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")



def main():
    global no_frame_file,file_dir
    # Load NIfTI images
    args = parse_arguments()
    print(f"Received input_file: {args.input_file}")
    input_file = args.input_file
    folder_name = os.path.splitext(os.path.basename(input_file))[0].replace('.nii', '')
    file_dir = os.path.dirname(input_file) + "/" + folder_name
    if not os.path.exists(file_dir):os.makedirs(file_dir)
    file_dir = file_dir + '/'
    no_frame_file = file_dir + "frame_" + os.path.basename(input_file.split('/')[-1])


    nii_img = nib.load(input_file)
    nii_img_data = nii_img.get_fdata()
    img_shape = nii_img_data.shape
    affine_matrix = nii_img.affine
    # Set NumPy output format options
    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)

    # Remove head shell
    slices = []
    for index in range(img_shape[2]):
        slice = nii_img_data[:, :, index]
        if index >= (img_shape[2]/4):
            frame = np.where(slice>97, 1, 0)
            frame = np.where(frame==1, 0, 1)
            slice = findstartm(frame, slice)
            blank = np.where(slice>20, 0, 1)
            slice = findstartm(blank, slice)
            tissue = np.where(slice>0, 1, 0)
            slice = clean(img_shape, slice, tissue)
            vessel = np.where(slice>95, slice, 0)
            slices.append(vessel)

    affine_matrix[0][0] = -affine_matrix[0][0]
    affine_matrix = affine_matrix[[1,2,0,3]]

    # vessel
    slices = np.array(slices)
    new_nii_img = nib.Nifti1Image(slices, affine_matrix)
    nib.save(new_nii_img, no_frame_file)
    no_frame_file = nifti_to_stl(new_nii_img)

    # Half brain
    half(no_frame_file)
    # ICP
    ICP(left_file, right_mirror)
    ICP(right_file, left_mirror)
    # KDtree Subtraction
    KDtree_subtraction(left_file, right_mirror)
    KDtree_subtraction(right_file, left_mirror)
    print(no_frame_file)

if __name__ == '__main__':
    main()