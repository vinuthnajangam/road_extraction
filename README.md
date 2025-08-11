# 🛰️ Road Network Extraction & Graph Construction from Satellite Imagery 🗺️

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Problem Statement & Motivation](#2-problem-statement--motivation)
- [3. Dataset](#3-dataset)
- [4. Methodology: An End-to-End Pipeline](#4-methodology-an-end-to-end-pipeline)
  - [4.1 Phase 1: Deep Learning-based Road Segmentation](#41-phase-1-deep-learning-based-road-segmentation)
  - [4.2 Phase 2: Post-Processing & Topological Graph Construction](#42-phase-2-post-processing--topological-graph-construction)
    - [4.2.1 Skeletonization (Raster Thinning)](#421-skeletonization-raster-thinning)
    - [4.2.2 Vectorization (Pixel Graph Traversal)](#422-vectorization-pixel-graph-traversal)
    - [4.2.3 Topological Correction](#423-topological-correction)
    - [4.2.4 Graph Construction](#424-graph-construction)
- [5. Results & Discussion](#5-results--discussion)
- [6. Challenges & Solutions](#6-challenges--solutions)
- [7. Key Learnings](#7-key-learnings)
- [8. Future Work & Extensions](#8-future-work--extensions)
- [9. Technical Stack](#9-technical-stack)
- [10. Setup & Usage](#10-setup--usage)

---

## 1. Project Overview

This project presents a robust, end-to-end pipeline for the **automatic extraction of road networks from high-resolution satellite imagery** and their transformation into **topologically accurate graph data structures**.

Instead of merely classifying pixels, this system provides a structured, usable graph of roads, complete with nodes (intersections, endpoints) and edges (road segments), offering a foundation for complex spatial analysis.

## 2. Problem Statement & Motivation

Accurate and up-to-date road network maps are fundamental to modern society. However, manual digitization of roads from satellite imagery is a labor-intensive, time-consuming, and error-prone process. Traditional image processing methods often struggle with the vast diversity of road appearances, occlusions, and varying image conditions.

This project aims to automate this process, addressing key challenges:
* **Semantic Understanding:** Precisely identifying road pixels amidst complex backgrounds.
* **Geometric Fidelity:** Converting amorphous pixel masks into clean, thin lines.
* **Topological Correctness:** Ensuring lines connect accurately at intersections and form a valid network graph, free from breaks, overshoots, or dangling spurs.
* **Scalability:** Laying the groundwork for processing large geographic areas.

## 3. Dataset

The model was trained and evaluated using the **DeepGlobe Road Extraction Challenge Dataset**.

* **Characteristics:** Consists of high-resolution satellite image tiles (256x256 pixels) with a ground sampling distance of approximately 0.5 meters per pixel. The dataset provides corresponding binary masks for road segmentation (training set).
* **Data Volume:** The training set comprises over 6,000 image-mask pairs, allowing for robust model training.
* **Data Preparation:** Images were preprocessed with resizing, normalization, and extensive data augmentation (flips, rotations, shifts, scale, brightness/contrast, hue/saturation/value) to enhance model generalization.

## 4. Methodology: An End-to-End Pipeline

This pipeline orchestrates a sequence of computer vision and graph processing techniques to achieve highly accurate road network extraction.

### 4.1 Phase 1: Deep Learning-based Road Segmentation

This phase focuses on leveraging deep learning to semantically understand and delineate road pixels.

* **Model Architecture:** A **U-Net** convolutional neural network was employed, known for its strong performance in semantic segmentation. The U-Net utilized a **pre-trained ResNet-34 encoder** (pre-trained on ImageNet).
    * *Why U-Net?* Its encoder-decoder structure with skip connections efficiently captures both high-level contextual information and precise spatial details, crucial for accurate pixel-wise road boundary detection.
    * *Why Pre-trained Encoder?* Transfer learning from ImageNet allowed the model to rapidly learn robust general image features, accelerating training and boosting performance.
* **Loss Function:** A composite loss function combining **Binary Cross-Entropy (BCE) Loss** and **Dice Loss** was used.
    * *Why Combined?* BCE provides stable optimization for pixel classification, while Dice Loss effectively addresses the severe class imbalance inherent in road segmentation (roads occupy only a small fraction of image pixels), ensuring the model prioritizes correct road detection.
* **Training:** The model was trained for 100 epochs with a learning rate of `1e-5` using the Adam optimizer. Extensive data augmentation was applied during training.
* **Performance:** Achieved a best validation IoU of **0.5325**, demonstrating high accuracy in pixel-level road detection on unseen data.

**Visual: Original Image & Predicted Mask**
![Predicted Mask Example](outputs/mask/just_mask_803789.png)

### 4.2 Phase 2: Post-Processing & Topological Graph Construction

This is the sophisticated algorithmic core that transforms raw pixel predictions into a structured graph.

#### 4.2.1 Skeletonization (Raster Thinning)

* **Objective:** Reduce the predicted road masks (thick pixel blobs) into single-pixel wide centerlines, which are essential for vectorization.
* **Method:** Utilized `skimage.morphology.skeletonize`, a robust algorithm that iteratively thins shapes while preserving their topological connectivity.

**Visual: Skeleton Mask & Skeleton Overlay**
![Skeletonization Example](outputs/skeleton/just_skeleton_803789.png)

#### 4.2.2 Vectorization (Pixel Graph Traversal)

* **Objective:** Convert the 1-pixel wide skeletonized mask into a list of `Shapely LineString` objects, representing vector road segments.
* **Method:** Implemented a custom **pixel graph traversal** algorithm.
    * A `NetworkX` graph is constructed where each active skeleton pixel is a node, and edges connect adjacent pixels.
    * The algorithm then systematically traces unique paths along these pixel-level connections, stopping at junctions or endpoints, ensuring each segment is vectorized exactly once.
    * This approach robustly handles complex skeleton topologies, avoiding issues encountered with traditional contour-finding methods (like `cv2.findContours`) that could yield degenerate or non-simple geometries.
* **Cleaning:** Generated `LineString`s are simplified using `line.simplify(tolerance=1.0)` to reduce jaggedness while preserving topology.

**Visual: Raw Vector Overlay**
![Raw Vector Overlay Example](outputs/vector/just_vectors_803789.png)

#### 4.2.3 Topological Correction

This is the critical phase where geometric imperfections are resolved, and lines are prepared for graph formation.

* **Node Identification:** All start and end points of `LineString`s are collected.
* **Node Snapping/Clustering:**
    * **Purpose:** To group spatially close points (endpoints, intersections) that logically represent the same location into a single, unique node.
    * **Method:** A custom iterative brute-force clustering algorithm (or KD-Tree for larger scales) is used, with a `tolerance` (e.g., 1.0-5.0 pixels) to snap nearby points. This step also provides a mapping from original points to their unique node IDs.
* **Line Connection:** `LineString` endpoints are precisely realigned to their corresponding snapped nodes (`connect_lines` function). This ensures topological correctness and allows the preservation of valid loops.
* **Line Splitting:** `LineString`s are accurately split at any internal intersection nodes they pass through (`split_lines` function), ensuring intersections are explicitly represented as graph nodes. This utilizes `line.difference(node.buffer())` for robust splitting.
* **Dangling Edge / Noisy Loop Removal:** (`clean_lines` function)
    * **Purpose:** Iteratively removes very short, isolated segments (spurs) or small, visually noisy loops that are artifacts.
    * **Method:** Builds a temporary `NetworkX` graph to check node degrees (connectivity). Removes lines shorter than `min_length` (e.g., 5-10 pixels) if they are dangling (degree 1 endpoint) or isolated. It also explicitly filters small, noisy loops (identified by `line.is_ring` and `line.area < max_noisy_loop_area`) regardless of their perimeter length.

**Visual: Cleaned Vector Overlay**
![Before Cleaning Overlay Example](outputs/vector/just_cleaned_vectors_2704.png)

#### 4.2.4 Graph Construction

* **Objective:** To formally represent the clean road network as a `NetworkX` graph.
* **Method:** An `nx.Graph()` object is created. Unique nodes (identified in snapping) are added, and their coordinates are stored as node attributes. Cleaned `LineString` objects are then added as edges between their corresponding snapped nodes, with attributes like `length` and the `Shapely geometry` itself. Robust node lookup using `KDTree` is incorporated for final edge creation.

## 5. Results & Discussion

The project successfully delivers a complete pipeline for automated road network extraction. The final `NetworkX` graph represents the road infrastructure with high fidelity.

* **Visual Quality:** The final network visualizations demonstrate clean, smoothed road segments accurately overlaid on the satellite imagery, with clearly identified intersections and endpoints.
* **Topological Correctness:** The graph is topologically sound, ensuring proper connectivity and an accurate representation of the road network for analysis and navigation.

**Visual: Final Network Overlay**
![Final Network Overlay 1](outputs/final/final_803789.png)
![Final Network Overlay 2](outputs/final/final_965066.png)
![Final Network Overlay 3](outputs/final/final_134034.png)
![Final Network Overlay 4](outputs/final/final_357084.png)
![Final Network Overlay 5](outputs/final/final_489439.png)
![Final Network Overlay 6](outputs/final/final_555827.png)
![Final Network Overlay 7](outputs/final/final_633197.png)
![Final Network Overlay 8](outputs/final/final_816042.png)

## 6. Challenges & Solutions

Developing this pipeline involved overcoming several complex challenges:

* **Overfitting in Deep Learning:** Addressed by extensive data augmentation and careful hyperparameter tuning.
* **`Shapely` & `OpenCV` Degeneracies:** Solved issues arising from `cv2.findContours` producing non-simple geometries (e.g., "turn-back" lines) and `Shapely`'s `buffer(0)` behavior through iterative debugging and the implementation of `line.buffer(0).boundary` and the pixel graph traversal method.
* **Floating-Point Precision:** Managed float precision issues in coordinate comparisons during node snapping and graph construction using explicit tolerances and KD-Trees.
* **Topological Complexity:** Successfully implemented iterative algorithms for node snapping, line splitting, and spur removal to achieve robust graph topology.
* **Pipeline Debugging:** The multi-stage nature required systematic debugging and visualization at each intermediate step to ensure correctness.

## 7. Key Learnings

This project provided invaluable experience in:
* Building and training deep learning models (U-Net) for semantic segmentation using PyTorch.
* Advanced image preprocessing and data augmentation with `Albumentations`.
* Low-level image processing with `OpenCV` and `scikit-image`.
* Mastering `Shapely` for complex geometric operations (buffering, simplification, intersection, difference).
* Implementing custom graph traversal algorithms (pixel graph).
* Understanding and enforcing geospatial topology.
* Graph data structures and network analysis with `NetworkX`.
* Systematic debugging of complex, multi-stage pipelines.

## 8. Future Work & Extensions

* **Large-Scale Processing:** Implement module for stitching multiple adjacent DeepGlobe tiles to generate graphs for larger geographical areas.
* **Road Attributes:** Extract additional attributes (e.g., road width, material, number of lanes) from imagery and add them as graph attributes.
* **Performance Optimization:** Explore GPU acceleration for post-processing steps or optimize algorithms for speed.
* **Deployment:** Create a user-friendly interface or API for automated mapping.

## 9. Technical Stack

* **Languages:** Python 3.x
* **Deep Learning Frameworks:** PyTorch
* **Computer Vision & Image Processing:** OpenCV, scikit-image, Albumentations
* **Geospatial & Geometry:** Shapely
* **Graph Theory:** NetworkX
* **Data Handling:** NumPy
* **Visualization:** Matplotlib, TQDM