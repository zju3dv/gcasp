import numpy as np
import open3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def points2pcd(points, colors = None, uni_colors = None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    if(colors is not None):
        pcd.colors = open3d.utility.Vector3dVector(colors)
    elif(uni_colors is not None):
        pcd.paint_uniform_color(uni_colors)

    return pcd

def draw_points(points, colors = None):
    pcds = []
    for i in range(points.shape[0]):
        pcd = open3d.geometry.TriangleMesh.create_sphere()
        pcd.compute_vertex_normals()
        if(colors is not None):
            pcd.paint_uniform_color(colors[i])
        pcd.scale(0.001,[0,0,0])
        pcd.translate(points[i])
        pcds.append(pcd)
    return pcds

def render_window(pcds):
    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window('img', width=800, height=600)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget)    


    mat1 = rendering.MaterialRecord()
    mat1.shader = 'defaultLit'
    mat1.base_color = [1,1,1,1]
    
    mat2 = rendering.MaterialRecord()
    mat2.shader = 'defaultLitTransparency'
    # widget.enable_sun_light(False)
    mat2.base_color = [1,0.5,0.5,0.5]

    widget.scene.camera.look_at([0,0,1], [0,0,-0.3], [0,1,0])
    # widget.scene.show_axes(True)
    
    for i,pcd in enumerate(pcds):
        widget.scene.add_geometry('ellipsoid{}'.format(i), pcd, mat1)

    gui.Application.instance.run()