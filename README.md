# The Cerebral Vascular Occlusion Rapid Detection and Localization System

This project is a medical auxiliary tool designed to assist healthcare professionals in diagnosing and treating cerebral vascular obstructions. By constructing a three-dimensional (3D) point cloud model of cerebral blood vessels from Computed Tomography Angiography (CTA) scans, the system identifies and marks vascular anomalies to provide clearer visualization and rapid localization of obstructions.

## Project Overview
Cerebral vascular disease is a leading cause of death and long-term disability in Taiwan. Current diagnostic processes for stroke often rely on manual interpretation of 2D images, which is time-consuming and requires high expertise. This system optimizes the interpretation of CTA scans to improve emergency triage efficiency and help patients receive treatment within the "golden window".

### Key Features
* **Technology Selection**: Uses Computed Tomography Angiography (CTA) because it is the first-line diagnostic tool in suspected stroke cases due to its speed compared to MRA.
* **3D Point Cloud Modeling**: Instead of traditional 2D slices or 3D voxels, the project utilizes **3D Point Cloud** technology. This provides better depth perception, enhances visual intuition, and reduces computational load while maintaining precise vessel structures.
* **Automated Detection**: Detects potential vascular anomalies by comparing the symmetry between the left and right hemispheres of the brain using point cloud distribution analysis.

## Technical Workflow
The system follows a structured processing pipeline:
1.  **Data Pre-processing**: Loads NIfTI files and uses a mask to filter out noise (such as the skull and low-intensity brain tissue) by removing voxels with values less than 85.
2.  **3D Model Construction**: Converts the processed medical images into a 3D point cloud model with a total of 8,000 points to balance model integrity and hardware performance.
3.  **Hemisphere Segmentation & Mapping**: Splits the model into left and right hemispheres based on the brain center and generates mirrored mapping files for alignment.
4.  **ICP Calibration & Comparison**: Uses the **Iterative Closest Point (ICP)** algorithm and KD-Tree method to align the hemispheres and identify points that lack symmetry as potential occlusion sites.
5.  **Visualization**: Marks the detected abnormal areas in red on the 3D model, providing a clear reference for medical staff.

## File & Data Architecture
The directory structure of the project is organized as follows:

```text
The-Cerebral-Vascular-Occlusion-Rapid-Detection-and-Localization-System-main/
├── Document/                 # Project documentation and reports
│   ├── 作品簡介.docx          # Project introduction and background summary
│   ├── 設計規格書.pdf          # Detailed system design specifications
│   ├── 需求規格書.pdf          # User requirements and functional planning
│   └── 測試報告書.pdf          # Testing records for accuracy and performance
├── IMP2023/                  # Academic publications (IMP Conference 2023)
│   ├── IMP-067_..._全文論文.pdf # Full research paper with experimental data
│   └── IMP-067_..._精簡論文.docx # Abridged paper focusing on core methodology
├── Final.py                  # Core algorithm script (Processing, ICP, KD-Tree comparison)
├── PV.py                     # GUI script (Point cloud viewer based on PyQt5 and PyVista)
├── Datalist                  # Record of datasets used (e.g., IACTA-EST 2023)
└── README.md                 # Project documentation
```

### Script Functionality
* **Final.py**: The backend logic for NIfTI processing, noise removal, 3D conversion, and hemisphere comparison.
* **PV.py**: The frontend Graphical User Interface (GUI) that allows users to select NIfTI files, trigger the processing pipeline, and visualize the resulting point cloud with marked anomalies.

### Environment Requirements
To run this project, the following Python libraries are required:
* `nibabel` (Handling NIfTI medical images)
* `open3d` (Point cloud processing and ICP algorithms)
* `pyvista` & `pyvistaqt` (3D visualization and plotting)
* `PyQt5` (Graphical User Interface)
* `numpy`, `scikit-image`, `scipy`, `pyntcloud`, `numpy-stl`
