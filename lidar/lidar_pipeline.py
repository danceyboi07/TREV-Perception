#VLP LiDAR Data Acquisition Draft
import numpy as np
from sensor_msgs_py import point_cloud2

def lidar_callback(msg):
    points = np.array([
        [p[0], p[1], p[2]]
        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        )
    ], dtype=float).reshape(-1, 3)

    return points

#RANSAC Ground Segmentation Draft

import numpy as np
import open3d as o3d

def remove_ground(points):
    if len(points) < 3:
        return points, np.empty((0, 3))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    plane, inliers = cloud.segment_plane(
        distance_threshold=0.08,
        ransac_n=3,
        num_iterations=100
    )

    # Avoid removing an obviously non-horizontal plane.
    normal = np.array(plane[:3])
    normal /= np.linalg.norm(normal)

    if abs(normal[2]) < 0.9:
        return points, np.empty((0, 3))

    ground = cloud.select_by_index(inliers)
    nonground = cloud.select_by_index(
        inliers,
        invert=True
    )

    return (
        np.asarray(nonground.points),
        np.asarray(ground.points)
    )
  
#DBSCAN Object Clustering Draft

from sklearn.cluster import DBSCAN

def cluster_cones(points):
    if len(points) == 0:
        return []

    labels = DBSCAN(
        eps=0.25,
        min_samples=3
    ).fit_predict(points)

    clusters = []

    for label in np.unique(labels):
        if label == -1:
            continue

        cluster = points[labels == label]
        clusters.append(cluster)

    return clusters

#Cone Reconstruction, and recovery of removed points Draft

import numpy as np

def reconstruct_cones(clusters, ground_points):
    reconstructed = []

    for cluster in clusters:
        center = cluster.mean(axis=0)

        if len(ground_points) == 0:
            reconstructed.append(cluster)
            continue

        # Horizontal distance from the cluster center
        distances = np.linalg.norm(
            ground_points[:, :2] - center[:2],
            axis=1
        )

        # Recover nearby removed points
        nearby = ground_points[
            distances < 0.15
        ]

        complete_cone = np.vstack([
            cluster,
            nearby
        ])

        reconstructed.append(complete_cone)

    return reconstructed

#Geometric Filtering - Cone Validation Draft

import numpy as np

def filter_cones(clusters):
    valid_cones = []

    for cluster in clusters:
        if len(cluster) < 3:
            continue

        dimensions = (
            cluster.max(axis=0) -
            cluster.min(axis=0)
        )

        width = max(
            dimensions[0],
            dimensions[1]
        )

        height = dimensions[2]

        if (
            0.05 < height < 0.60
            and 0.03 < width < 0.40
        ):
            valid_cones.append(cluster)

    return valid_cones

#Cone localization and 3D Coordinates Estimation Draft

def estimate_cone_positions(cones):
    positions = []

    for cone in cones:
        center = cone.mean(axis=0)
        positions.append(center)

    return np.array(positions).reshape(-1, 3)

#Total Pipeline Order

def process_lidar(points):

    # Stage 1: RANSAC ground removal
    nonground, ground = remove_ground(points)

    # Stage 2: DBSCAN clustering
    clusters = cluster_cones(nonground)

    # Stage 3: Recover removed cone points
    reconstructed = reconstruct_cones(
        clusters,
        ground
    )

    # Stage 4: Geometric validation
    cones = filter_cones(reconstructed)

    # Stage 5: Estimate cone coordinates
    positions = estimate_cone_positions(cones)

    return positions
