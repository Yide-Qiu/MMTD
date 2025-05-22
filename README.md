
# MMM-TTA: Multi-Probe Benchmark for Maritime Multi-Object Tracking and Trajectory Association

[![Paper](https://img.shields.io/badge/arXiv-Paper-31AE8A)]()
[![Dataset](https://img.shields.io/badge/Docs-MMTDS_Dataset-0089D6)](https://yide-qiu.github.io/Pages_MMMTAA_Dataset/)
[![Download Dataset](https://img.shields.io/badge/Download-Data-10B981)](https://pan.quark.cn/s/42ff735ebab5)
[![Code](https://img.shields.io/badge/Code-Data-10B981)](https://github.com/Yide-Qiu/MMTD/)

Official implementation of the first dynamic observation heterogeneous constellation-based maritime multi-object tracking benchmark.

## Key Contributions

### 🌟 **MMM-TSS Simulation System**
- First dynamic observation constellation-based trajectory simulation system
- Implements **Walker-Integrated probe groups** (128 medium-altitude + 72 low-altitude probes)
- Features adaptive spatiotemporal registration and multi-probe resampling
- Supports radar (0-800m precision) and electronic reconnaissance (0-10km precision) sensors

### 🚢 **MMTDS Dataset Suite**
| Dataset        | Objects | Trajectory Points | Sensor Ratio (Radar/ER) | Resolution | 
|----------------|---------|-------------------|-------------------------|------------|
| MMTDS-sim      | 17,967  | 5.0M GT / 9.9M ER | 2.67% / 99.96%          | 100m       |
| MMTDS-ais      | 7,970   | 10.5M mixed       | 12.71% / 99.98%         | Real-world |
| MMTDS-[6 others]| 1,256-8,002 | 25K-505K    | 7-35% / 71-97%       | 0.92m-1km  |


@article{anonymous2024mmmtta,
  title={MMM-TTA: Multi-Probe Benchmark for Maritime Multi-Object Tracking},
  author={Anonymous},
  journal={NeurIPS},
  year={2025}
}
