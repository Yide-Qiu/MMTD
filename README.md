
# MMM-TTA: Multi-Probe Benchmark for Maritime Multi-Object Tracking and Trajectory Association

[![Paper](https://img.shields.io/badge/arXiv-Paper-31AE8A)]()
[![Dataset](https://img.shields.io/badge/Docs-MMTDS_Dataset-0089D6)](https://yide-qiu.github.io/Pages_MMMTAA_Dataset/)
[![Download Dataset](https://img.shields.io/badge/Download-Data-10B981)](https://pan.quark.cn/s/42ff735ebab5)
[![Code](https://img.shields.io/badge/Code-Data-0089D6)](https://github.com/Yide-Qiu/MMTD/)

Official implementation of the first dynamic observation heterogeneous constellation-based maritime multi-object tracking benchmark.

## Key Contributions

### 🌟 MMM-TSS Simulation System
- First dynamic observation constellation-based trajectory simulation system
- Implements **Walker-Integrated probe groups** (128 medium-altitude + 72 low-altitude probes)
- Features adaptive spatiotemporal registration and multi-probe resampling
- Supports radar (0-800m precision) and electronic reconnaissance (0-10km precision) sensors

### 🚢 MMTDS Dataset Suite
| Dataset        | Objects | Trajectory Points | Sensor Ratio (Radar/ER) | Resolution | 
|----------------|---------|-------------------|-------------------------|------------|
| MMTDS-sim      | 17,967  | 5.0M GT / 9.9M ER | 2.67% / 99.96%          | 100m       |
| MMTDS-ais      | 7,970   | 10.5M mixed       | 12.71% / 99.98%         | Real-world |
| MMTDS-[6 others]| 1,256-8,002 | 25K-505K    | 7-35% / 71-97%       | 0.92m-1km  |

### 🏆 Benchmark Performance
| Model       | MMTDS-sim (HOTA/DetA/AssA/IDF1)            | MMTDS-satmtb (HOTA/DetA/AssA/IDF1)         | MMTDS-ootb (HOTA/DetA/AssA/IDF1)           | MMTDS-otb100 (HOTA/DetA/AssA/IDF1)         |
|-------------|--------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
| ByteTrack   | 17.80±0.23 | 23.72±0.19 | 13.42±0.31 | 17.65±0.27 | 25.14±0.18 | 25.99±0.22 | 24.50±0.29 | 41.55±0.33 | 33.05±0.28 | 21.55±0.24 | 50.97±0.37 | 35.07±0.26 | 41.30±0.31 | 30.24±0.23 | 57.13±0.42 | 46.50±0.39 |
| OCSORT      | 10.80±0.18 | 20.96±0.21 | 5.60±0.12  | 10.50±0.15 | 25.56±0.27 | 29.61±0.32 | 22.24±0.25 | 42.07±0.36 | 36.75±0.29 | 25.87±0.23 | 52.73±0.41 | 35.99±0.33 | 36.30±0.28 | 31.95±0.26 | 42.29±0.34 | 36.48±0.31 |
| HybridSORT  | 13.98±0.22 | 23.23±0.25 | 8.46±0.17  | 15.33±0.19 | 24.87±0.24 | 27.31±0.28 | 22.86±0.27 | 39.47±0.35 | 37.42±0.32 | 26.47±0.29 | 53.62±0.43 | 37.02±0.34 | 42.23±0.37 | 37.55±0.33 | 48.85±0.42 | 41.55±0.38 |
| UCMCTrack   | 17.27±0.21 | 24.33±0.26 | 12.32±0.19 | 18.43±0.22 | 18.72±0.17 | 26.56±0.24 | 13.22±0.18 | 39.45±0.34 | 28.80±0.23 | 22.49±0.21 | 37.03±0.31 | 34.55±0.29 | 31.00±0.27 | 30.95±0.25 | 31.17±0.28 | 42.80±0.36 |
| DiffMOT     | 24.02±0.29 | 26.16±0.31 | 22.14±0.28 | 26.98±0.33 | 25.06±0.26 | 27.85±0.29 | 22.57±0.27 | 38.84±0.35 | 38.92±0.34 | 22.81±0.23 | 66.48±0.53 | 44.27±0.41 | 44.42±0.39 | 30.52±0.28 | 64.66±0.51 | 55.90±0.47 |
| **MIMMA**   | 33.74±0.32 | 24.44±0.27 | 46.70±0.41 | 42.94±0.38 | 44.46±0.39 | 48.13±0.43 | 42.44±0.37 | 54.79±0.48 | 49.03±0.42 | 26.52±0.28 | 92.98±0.72 | 50.00±0.45 | 55.04±0.49 | 39.43±0.35 | 76.89±0.63 | 66.67±0.58 |

| Model       | MMTDS-satsot (HOTA/DetA/AssA/IDF1)         | MMTDS-viso (HOTA/DetA/AssA/IDF1)           | MMTDS-mtad (HOTA/DetA/AssA/IDF1)           | MMTDS-ais (HOTA/DetA/AssA/IDF1)            |
|-------------|--------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
| ByteTrack   | 32.01±0.31 | 24.78±0.23 | 41.45±0.37 | 41.55±0.36 | 28.26±0.27 | 14.21±0.16 | 56.22±0.49 | 25.64±0.24 | 39.78±0.35 | 39.63±0.34 | 40.04±0.36 | 54.81±0.47 | 19.36±0.21 | 27.07±0.25 | 13.87±0.17 | 21.18±0.22 |
| OCSORT      | 41.68±0.38 | 30.54±0.29 | 57.01±0.51 | 44.37±0.40 | 27.09±0.26 | 14.45±0.15 | 50.78±0.45 | 23.89±0.23 | 30.43±0.28 | 38.93±0.35 | 23.89±0.24 | 43.89±0.39 | 14.86±0.16 | 31.68±0.28 | 6.98±0.09  | 26.61±0.25 |
| HybridSORT  | 42.76±0.39 | 32.23±0.31 | 56.97±0.52 | 44.85±0.41 | 27.96±0.26 | 14.45±0.14 | 54.09±0.48 | 24.25±0.22 | 24.03±0.23 | 31.96±0.29 | 18.17±0.19 | 33.92±0.31 | 14.25±0.15 | 30.33±0.27 | 6.70±0.08  | 25.07±0.24 |
| UCMCTrack   | 27.47±0.25 | 24.98±0.23 | 30.24±0.28 | 38.60±0.35 | 23.38±0.22 | 13.57±0.14 | 40.33±0.36 | 24.75±0.23 | 29.09±0.27 | 39.54±0.36 | 21.50±0.21 | 40.96±0.37 | 11.03±0.12 | 29.60±0.27 | 4.11±0.06  | 14.54±0.16 |
| DiffMOT     | 42.16±0.38 | 25.71±0.24 | 69.20±0.61 | 49.72±0.44 | 22.48±0.21 | 15.47±0.16 | 32.71±0.30 | 29.08±0.27 | 27.91±0.26 | 38.82±0.35 | 20.18±0.20 | 35.76±0.32 | 8.19±0.09  | 25.78±0.24 | 2.61±0.04  | 7.62±0.08  |
| **MIMMA**   | 42.37±0.39 | 18.95±0.18 | 94.74±0.83 | 33.33±0.30 | 61.95±0.55 | 72.49±0.67 | 53.08±0.47 | 73.10±0.65 | 40.73±0.37 | 42.44±0.38 | 39.33±0.35 | 53.43±0.48 | 22.74±0.23 | 31.77±0.29 | 16.31±0.18 | 30.18±0.28 |


### Citation
@article{anonymous2025mmmtta,<br>
  title     = {MMM-TTA: Multi-Probe Benchmark for Maritime Multi-Object Tracking},<br>
  author    = {Anonymous},<br>
  journal   = {NeurIPS},<br>
  year      = {2025},<br>
  note      = {Under Review}<br>
}
