import open3d as o3d
import numpy as np

# 读取XYZ文件
points = np.loadtxt("output_point_cloud.xyz")
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 显示点云
o3d.visualization.draw_geometries([point_cloud])
