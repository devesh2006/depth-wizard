import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from models.depth import PointCloudData, MeshHeightfieldData

def reconstruct_3d_geometry(
    img: Image.Image,
    depth: np.ndarray,
    scale_factor: float = 1.0,
    intrinsics: Optional[Dict[str, float]] = None,
    grid_res: int = 120,
    pt_sample_step: int = 4
) -> Tuple[PointCloudData, MeshHeightfieldData]:
    """
    Back-projects depth map and RGB image to 3D Point Cloud and Heightfield Mesh.
    Uses pinhole camera back-projection geometry:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth * scale_factor
    """
    # Geometry is indexed in DEPTH-GRID space, so the principal point and the
    # normalisation must use the depth grid dimensions. Using the original image
    # size here (which can be 4x larger) drove every (u - cx) term negative and
    # pushed the whole point cloud off to one corner.
    h, w = depth.shape[0], depth.shape[1]
    
    # 1. Camera Intrinsics
    if intrinsics is None:
        # Default approximate pinhole camera intrinsics
        fx = float(intrinsics.get("fx", max(w, h) * 1.2)) if intrinsics else float(max(w, h) * 1.2)
        fy = float(intrinsics.get("fy", max(w, h) * 1.2)) if intrinsics else float(max(w, h) * 1.2)
        cx = float(intrinsics.get("cx", w / 2.0)) if intrinsics else float(w / 2.0)
        cy = float(intrinsics.get("cy", h / 2.0)) if intrinsics else float(h / 2.0)
    else:
        fx = float(intrinsics.get("fx", max(w, h) * 1.2))
        fy = float(intrinsics.get("fy", max(w, h) * 1.2))
        cx = float(intrinsics.get("cx", w / 2.0))
        cy = float(intrinsics.get("cy", h / 2.0))
        
    # Resize RGB image to match depth map if necessary
    img_rgb = img.resize((depth.shape[1], depth.shape[0]), Image.Resampling.BILINEAR).convert('RGB')
    rgb_arr = np.array(img_rgb, dtype=np.float32) / 255.0
    
    # 2. Generate Heightfield Mesh Grid (e.g. 120x120 for smooth real-time WebGL mesh)
    mesh_w = grid_res
    mesh_h = int(grid_res * (h / w))
    
    # Sample depth and colors for mesh
    y_mesh_idx = np.linspace(0, depth.shape[0] - 1, mesh_h).astype(int)
    x_mesh_idx = np.linspace(0, depth.shape[1] - 1, mesh_w).astype(int)
    
    mesh_depth = depth[np.ix_(y_mesh_idx, x_mesh_idx)]
    mesh_colors = rgb_arr[np.ix_(y_mesh_idx, x_mesh_idx)]
    
    # Height elevation with scale factor
    mesh_heights = (mesh_depth * scale_factor).flatten().tolist()
    mesh_colors_flat = mesh_colors.reshape(-1, 3).flatten().tolist()
    
    mesh_data = MeshHeightfieldData(
        grid_width=mesh_w,
        grid_height=mesh_h,
        heights=[round(float(v), 4) for v in mesh_heights],
        colors=[round(float(v), 3) for v in mesh_colors_flat]
    )
    
    # 3. Generate Point Cloud using Pinhole Camera Back-Projection
    step_y = max(1, depth.shape[0] // 160)
    step_x = max(1, depth.shape[1] // 160)
    
    y_pts = np.arange(0, depth.shape[0], step_y)
    x_pts = np.arange(0, depth.shape[1], step_x)
    
    xx, yy = np.meshgrid(x_pts, y_pts)
    zz = depth[yy, xx]  # metric depth in metres (Z_cam)
    
    # Check if depth is in metres (values > 1.0) or normalized relative
    is_metric_scale = bool(np.mean(zz) > 1.0)
    if is_metric_scale:
        # True Pinhole camera back-projection: X = (u - cx) * Z / fx, Y = -(v - cy) * Z / fy
        cam_x = (xx - cx) * zz / fx
        cam_y = -(yy - cy) * zz / fy
        cam_z = zz
    else:
        cam_x = (xx - cx) / (w / 2.0) * 50.0
        cam_y = -(yy - cy) / (h / 2.0) * 50.0
        cam_z = zz * 30.0 * scale_factor
    
    pts_colors = rgb_arr[yy, xx]
    
    pos_flat = np.stack([cam_x, cam_y, cam_z], axis=-1).reshape(-1, 3)
    col_flat = pts_colors.reshape(-1, 3)
    
    # Outlier filtering
    median_z = np.median(pos_flat[:, 2])
    std_z = np.std(pos_flat[:, 2])
    valid_mask = np.abs(pos_flat[:, 2] - median_z) < (4.0 * std_z + 1e-3)
    
    valid_pos = pos_flat[valid_mask]
    valid_col = col_flat[valid_mask]
    
    min_x, max_x = float(np.min(valid_pos[:, 0])), float(np.max(valid_pos[:, 0]))
    min_y, max_y = float(np.min(valid_pos[:, 1])), float(np.max(valid_pos[:, 1]))
    min_z, max_z = float(np.min(valid_pos[:, 2])), float(np.max(valid_pos[:, 2]))
    
    total_pts = len(valid_pos)
    if total_pts > 20000:
        sub_indices = np.linspace(0, total_pts - 1, 20000).astype(int)
        valid_pos = valid_pos[sub_indices]
        valid_col = valid_col[sub_indices]
        
    pointcloud_data = PointCloudData(
        points_count=len(valid_pos),
        positions=[round(float(v), 3) for v in valid_pos.flatten()],
        colors=[round(float(v), 3) for v in valid_col.flatten()],
        bounds={
            "min_x": round(min_x, 2), "max_x": round(max_x, 2),
            "min_y": round(min_y, 2), "max_y": round(max_y, 2),
            "min_z": round(min_z, 2), "max_z": round(max_z, 2),
        }
    )
    
    return pointcloud_data, mesh_data

def export_pointcloud_ply(positions: np.ndarray, colors: np.ndarray) -> str:
    """Generates ASCII PLY format point cloud string."""
    num_pts = len(positions)
    header = f"""ply
format ascii 1.0
comment DepthWizard SIH26175 3D Point Cloud
element vertex {num_pts}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    lines = [header]
    for p, c in zip(positions, colors):
        r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
        lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")
    return "".join(lines)

def export_mesh_obj(mesh_data: MeshHeightfieldData) -> str:
    """Generates Wavefront OBJ mesh file from heightfield grid."""
    gw = mesh_data.grid_width
    gh = mesh_data.grid_height
    heights = np.array(mesh_data.heights).reshape((gh, gw))
    
    lines = ["# DepthWizard SIH26175 3D Height Mesh\n"]
    
    # Vertices
    for r in range(gh):
        for c in range(gw):
            x = (c / (gw - 1) - 0.5) * 100.0
            z = -(r / (gh - 1) - 0.5) * 100.0
            y = heights[r, c] * 25.0
            lines.append(f"v {x:.3f} {y:.3f} {z:.3f}\n")
            
    # Faces (quads split into two triangles)
    for r in range(gh - 1):
        for c in range(gw - 1):
            # 1-based indexing in OBJ
            v1 = r * gw + c + 1
            v2 = r * gw + (c + 1) + 1
            v3 = (r + 1) * gw + (c + 1) + 1
            v4 = (r + 1) * gw + c + 1
            lines.append(f"f {v1} {v2} {v3}\n")
            lines.append(f"f {v1} {v3} {v4}\n")
            
    return "".join(lines)
