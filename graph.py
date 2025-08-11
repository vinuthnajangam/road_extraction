import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage.morphology as sk
from shapely.geometry import LineString, Point, MultiLineString, GeometryCollection, Polygon
import networkx as nx
import random
from scipy.spatial import KDTree

from model import UNetWithPretrainedEncoder
from utils import load_checkpoint
import config


def get_test_images(test_dir, num_images=5):
    all_images = sorted(os.listdir(test_dir))
    
    if len(all_images) > num_images:
        test_images_names = random.sample(all_images, num_images)
    else:
        test_images_names = all_images
        
    test_images = []
    for img_name in test_images_names:
        img_path = os.path.join(test_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_images.append((image, img_name))
        
    return test_images

def get_particular_images(test_dir, nums):
    
    test_images_paths = [os.path.join(test_dir, f"{num}_sat.jpg") for num in nums]
    
    test_images = []
    for img_path in test_images_paths:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_images.append((image, img_path.split("_")[0]))
        
    return test_images
    

def skel(model, original_image, transform, device):
    transformed = transform(image=original_image)
    input_image_tensor = transformed['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_image_tensor)
        
    predicted_mask_tensor = torch.sigmoid(logits)
    predicted_mask_tensor = (predicted_mask_tensor > 0.5).float()

    predicted_mask_np = predicted_mask_tensor.squeeze().cpu().numpy()
    
    mask_bool = predicted_mask_np.astype(bool)
    skeleton = sk.skeletonize(mask_bool)
    skeleton_float = skeleton.astype(np.float32)
    print(skeleton_float.shape)
    return skeleton_float    

def vectorize_mask(skeleton):
    vector_lines = []
    
    skeleton_bool = skeleton.astype(bool)
    height, width = skeleton.shape

    pixel_graph = nx.Graph()

    neighbors_offsets = [
        (-1, -1), (-1, 0), (-1, 1), 
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1) 
    ]

    pixel_coord_to_id = {}
    id_counter = 0
    
    active_pixels_coords = np.argwhere(skeleton_bool)
    
    for r, c in active_pixels_coords:
        pixel_graph.add_node(id_counter, coords=(r, c))
        pixel_coord_to_id[(r, c)] = id_counter
        id_counter += 1

    for r, c in active_pixels_coords:
        current_pixel_id = pixel_coord_to_id[(r, c)]
        for dr, dc in neighbors_offsets:
            nr, nc = r + dr, c + dc 
            if 0 <= nr < height and 0 <= nc < width and skeleton_bool[nr, nc]:
                neighbor_pixel_id = pixel_coord_to_id[(nr, nc)]
                if not pixel_graph.has_edge(current_pixel_id, neighbor_pixel_id):
                    pixel_graph.add_edge(current_pixel_id, neighbor_pixel_id)

    traced_edges = set() 
    
    for u_start_id, v_start_id in pixel_graph.edges():
        if (u_start_id, v_start_id) in traced_edges or (v_start_id, u_start_id) in traced_edges:
            continue
        
        current_path_node_ids = [u_start_id, v_start_id]
        
        traced_edges.add(tuple(sorted((u_start_id, v_start_id)))) 
        
        current_head_id = v_start_id
        previous_node_in_path = u_start_id
        
        while True:
            next_possible_nodes = [
                n for n in pixel_graph.neighbors(current_head_id) 
                if n != previous_node_in_path
            ]
            
            next_untraced_nodes = []
            for next_n_id in next_possible_nodes:
                if tuple(sorted((current_head_id, next_n_id))) not in traced_edges:
                    next_untraced_nodes.append(next_n_id)

            if len(next_untraced_nodes) == 1: 
                next_node_id = next_untraced_nodes[0]
                
                current_path_node_ids.append(next_node_id)
                traced_edges.add(tuple(sorted((current_head_id, next_node_id))))
                
                previous_node_in_path = current_head_id
                current_head_id = next_node_id
                
            else: 
                break 
        
        segment_coords_rc = [pixel_graph.nodes[n_id]['coords'] for n_id in current_path_node_ids]
        
        if len(segment_coords_rc) >= 2:
            xy_coords = [(c, r) for r, c in segment_coords_rc] 
            
            line = LineString(xy_coords)
            
            simplified_line = line.simplify(tolerance=1.0, preserve_topology=True)
            
            if not simplified_line.is_empty:
                vector_lines.append(simplified_line)
    
    return vector_lines

def initial_nodes(lines):
    initial_nodes = []
    for line in lines:
        initial_nodes.append(line.coords[0])
        initial_nodes.append(line.coords[-1])
                
    final = np.array(initial_nodes)
    return final 

def snapping_nodes(init_nodes, tolerance):
    final_nodes = [] 
    map = {}
    visited = set() 
    node_counter = 0      

    init_nodes_tuples = [tuple(p) for p in init_nodes]

    for i in range(len(init_nodes)):
        if i in visited:
            continue

        cur_cluster_np = [init_nodes[i]] 
        cur_cluster = [init_nodes_tuples[i]] 
        visited.add(i)

        for j in range(i + 1, len(init_nodes)):
            if j not in visited: 
                distance = np.linalg.norm(init_nodes[i] - init_nodes[j])

                if distance <= tolerance:
                    cur_cluster_np.append(init_nodes[j])
                    cur_cluster.append(init_nodes_tuples[j])
                    visited.add(j) 

        centroid_np = np.mean(cur_cluster_np, axis=0)
        centroid_tuple = tuple(centroid_np) 

        node_id = node_counter
        final_nodes.append(centroid_tuple)

        for coord in cur_cluster:
            map[coord] = (node_id, centroid_tuple)
            
        node_counter += 1 
    
    final_nodes_array = np.array(final_nodes)
    
    return final_nodes_array, map

def connect_lines(lines, map, min_length): 
    connected_lines = []
    
    for line in lines:
        start = tuple(line.coords[0])
        end = tuple(line.coords[-1])
        
        start_node = map.get(start)
        end_node = map.get(end)
        
        if start_node is not None and end_node is not None:
            start_coord = start_node[1] 
            end_coord = end_node[1]     

            if start_coord == end_coord and line.length < min_length:
                continue

            intermediate_coords = list(line.coords)[1:-1] 

            new_line_coords = [list(start_coord)] + [list(coord) for coord in intermediate_coords] + [list(end_coord)]
            
            new_line = LineString(new_line_coords)
            
            if not new_line.is_empty:
                connected_lines.append(new_line)
            
    return connected_lines

def split_lines(connected_lines, final_nodes, tolerance=1e-6, buffer_radius_for_cut=0.1):
    final_lines = []
    
    node_points = [Point(coord) for coord in final_nodes]

    lines_to_process = list(connected_lines)
    
    while lines_to_process:
        current_line = lines_to_process.pop(0)
        
        if current_line.is_empty or not isinstance(current_line, LineString):
            continue
        
        was_split = False 
        
        for node_point in node_points:
            is_endpoint = (node_point.distance(Point(current_line.coords[0])) < tolerance or
                node_point.distance(Point(current_line.coords[-1])) < tolerance)

            if current_line.intersects(node_point) and not is_endpoint:
                
                cut_result = current_line.difference(node_point.buffer(buffer_radius_for_cut))
                
                if isinstance(cut_result, (MultiLineString, GeometryCollection)):
                    for segment in cut_result.geoms:
                        if isinstance(segment, LineString) and not segment.is_empty:
                            lines_to_process.append(segment)
                    was_split = True
                    break 
                elif isinstance(cut_result, LineString) and not cut_result.is_empty:
                    lines_to_process.append(cut_result)
                    was_split = True
                    break
        
        if not was_split:
            final_lines.append(current_line)
            
    return final_lines

def visualize_results(original_image, skeleton, raw_vectors, vector_lines, final_nodes, G, title="Result"):
    
    original_height, original_width, _ = original_image.shape
    model_size = 256

    scale_x = original_width / model_size
    scale_y = original_height / model_size
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 7))
    fig.suptitle(title, fontsize=18)

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    overlay_skel_image = original_image.copy()
    overlay_skel_image[skeleton == 1] = (
        overlay_skel_image[skeleton == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    axes[3].imshow(overlay_skel_image)
    axes[3].set_title('Skeleton Overlay')
    axes[3].axis('off')
    
    axes[4].imshow(original_image)
    axes[4].set_title('Raw Vector Overlay')
    for line in raw_vectors:
        if line.is_empty: 
            continue
        coords = np.array(line.coords)
        scaled_x_coords = coords[:, 0] * scale_x
        scaled_y_coords = coords[:, 1] * scale_y
        axes[4].plot(scaled_x_coords, scaled_y_coords, color='blue', linewidth=1) 
    #for nodeB in final_nodes:
     #   axes[2].plot(nodeB[0] * scale_x, nodeB[1] * scale_y, marker='o', color='red', markersize=2)
    axes[4].set_aspect('equal')
    axes[4].axis('off')
    
    
    axes[2].imshow(original_image)
    axes[2].set_title('Cleaned Vector Overlay')
    for line in vector_lines:
        if line.is_empty: 
            continue
        coords = np.array(line.coords)
        scaled_x_coords = coords[:, 0] * scale_x
        scaled_y_coords = coords[:, 1] * scale_y
        axes[2].plot(scaled_x_coords, scaled_y_coords, color='blue', linewidth=1) 
    #for nodeB in final_nodes:
     #   axes[3].plot(nodeB[0] * scale_x, nodeB[1] * scale_y, marker='o', color='red', markersize=2)
    axes[2].set_aspect('equal')
    axes[2].axis('off')
    
    axes[1].imshow(original_image)
    axes[1].set_title('Final Network Overlay')
    for u, v, data in G.edges(data=True):
        line_geometry = data['geometry']
        
        coords = np.array(line_geometry.coords)
        scaled_x_coords = coords[:, 0] * scale_x
        scaled_y_coords = coords[:, 1] * scale_y
        
        axes[1].plot(scaled_x_coords, scaled_y_coords, color='red', linewidth=2, zorder=2) 

    for node_id, data in G.nodes(data=True):
        node_x = data['x']
        node_y = data['y']
        
        scaled_node_x = node_x * scale_x
        scaled_node_y = node_y * scale_y
        
        degree = G.degree(node_id)
        if degree == 1: 
            node_color = 'blue'
            marker_size = 5
        elif degree > 2: 
            node_color = 'green'
            marker_size = 8
        else: 
            node_color = 'white'
            marker_size = 0
        
        axes[1].plot(scaled_node_x, scaled_node_y, marker='o', color=node_color, markersize=marker_size, zorder=3)
    axes[1].axis('off')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def clean_lines(lines, map, min_length):
    current_lines = list(lines) 
    
    new_map = {val[1]: val[0] for val in map.values()}
    
    removed_a_line = True 
    
    max_noisy_loop_area = 50.0 
    
    while removed_a_line:
        removed_a_line = False 
        temp_graph = nx.Graph()
        
        for node_id, _ in map.values(): 
            temp_graph.add_node(node_id)
        
        for line in current_lines:
            if line.is_empty: 
                continue
            
            start = tuple(line.coords[0])
            end = tuple(line.coords[-1])
            
            u_node_id = new_map.get(start)
            v_node_id = new_map.get(end)
            
            if u_node_id is not None and v_node_id is not None and u_node_id != v_node_id: 
                temp_graph.add_edge(u_node_id, v_node_id, geometry=line) 
            
        next_iteration_lines = []
        
        for line in current_lines:
            if line.is_empty:
                continue 
            
            start = tuple(line.coords[0])
            end = tuple(line.coords[-1])
                
            u_node_id = new_map.get(start)
            v_node_id = new_map.get(end)

            if u_node_id is None or v_node_id is None:
                removed_a_line = True
                continue
            
            if u_node_id == v_node_id:
                if line.is_ring and line.area < max_noisy_loop_area: 
                    removed_a_line = True
                    continue 

            if line.length < min_length:
                u_degree = temp_graph.degree(u_node_id)
                v_degree = temp_graph.degree(v_node_id)
                
                if (u_degree == 1 and v_degree > 1) or (v_degree == 1 and u_degree > 1): 
                    removed_a_line = True
                    continue 
                elif u_degree == 1 and v_degree == 1:
                    removed_a_line = True
                    continue 
                
            next_iteration_lines.append(line) 
        
        current_lines = next_iteration_lines
        
    return current_lines

def graph(final_lines, final_nodes, map):
    G = nx.Graph()
    
    new_map = {val[1]: val[0] for val in map.values()}
    
    node_coords_kdtree = KDTree(final_nodes)
    
    
    for rep_coord_tuple in new_map:
        node_id = new_map[rep_coord_tuple]
        G.add_node(node_id, x=rep_coord_tuple[0], y=rep_coord_tuple[1])
            
    node_match_tolerance = 1e-6 
    
    for line in final_lines: 
        if line.is_empty: 
            continue

        start_line_coord = tuple(line.coords[0]) 
        end_line_coord = tuple(line.coords[-1]) 
        
        dist_u, idx_u = node_coords_kdtree.query(start_line_coord, k=1)
        dist_v, idx_v = node_coords_kdtree.query(end_line_coord, k=1)

        u_node_id = None
        v_node_id = None

        if dist_u < node_match_tolerance:
            closest_rep_coord_u = tuple(final_nodes[idx_u])
            u_node_id = new_map.get(closest_rep_coord_u)

        if dist_v < node_match_tolerance:
            closest_rep_coord_v = tuple(final_nodes[idx_v])
            v_node_id = new_map.get(closest_rep_coord_v)

        if u_node_id is not None and v_node_id is not None and u_node_id != v_node_id:
            if not G.has_edge(u_node_id, v_node_id): 
                G.add_edge(u_node_id, v_node_id, 
                           length=line.length, 
                           geometry=line,
                           u_coords=start_line_coord, 
                           v_coords=end_line_coord)
            
    return G

def main():
    model = UNetWithPretrainedEncoder(encoder_name='resnet34', num_classes=1)
    
    checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    optimizer = torch.optim.Adam(model.parameters()) 
    load_checkpoint(checkpoint_path, model, optimizer)
    
    device = config.DEVICE
    model.to(device)
    model.eval()
    
    test_transform = config.val_transforms
    
    nums = [803789, 965066, 816042, 555827, 134034, 489439, 633197, 357084]
    
    print("Loading test images...")
    test_images_data = get_particular_images(test_dir=config.TEST_DIR, nums=nums)
    
    if not test_images_data:
        print("No images found in the test directory.")
        return
    
    min_length = 40.0

    print("Generating and visualizing masks...")
    for original_image, filename in tqdm(test_images_data, desc="Processing images"):
        mask_256 = skel(model, original_image, test_transform, device)
        vector_lines = vectorize_mask(mask_256)
        init_nodes = initial_nodes(vector_lines)
        mid_nodes, node_line_map = snapping_nodes(init_nodes, tolerance=10.0)
        connected_lines = connect_lines(vector_lines, node_line_map, min_length)    
        
        split = split_lines(connected_lines, mid_nodes)
        final_lines = clean_lines(split, node_line_map, min_length)
        final_nodes = initial_nodes(final_lines)
        
        G = graph(final_lines, mid_nodes, node_line_map)

        mask_resized = cv2.resize(mask_256, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        
        visualize_results(original_image, mask_resized, vector_lines, final_lines, final_nodes,G,title=f"Image: {filename}")

if __name__ == "__main__":
    main()