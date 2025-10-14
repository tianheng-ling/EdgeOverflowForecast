# EdgeOverflowForecast
![Time-Series](https://img.shields.io/badge/Time--Series-Combined%20Sewer%20System-orange) ![FPGA](https://img.shields.io/badge/FPGA-Optimized-blue) ![Quantization](https://img.shields.io/badge/Quantized-LSTM%20%7C%20Transformer-green)

### Overview

**EdgeOverflowForecast** is an end-to-end framework for **integer-only quantized deep learning models** (LSTM and Transformer) targeting **low-power embedded FPGAs**.  
It is designed for **short-term filling level forecasting in combined sewer systems** to support energy-efficient and resilient smart city infrastructure.

This repository provides:
- 🔧 Support **training and quantization-aware training (QAT)** of LSTM and Transformer models  
- 🧱 Automated **RTL code generation and FPGA synthesis** for deployment  
- ⚙️ Integrated **hardware-aware hyperparameter optimization** using Optuna  

> ⚠️ This repository works in tandem with our [ElasticAI.Creator](https://github.com/es-ude/elastic-ai.creator) library, which provides the core VHDL templates and quantization modules for hardware generation. Please make sure to install it as part of the setup process.

---

### Corresponding Paper

**"Automated Energy-Aware Time-Series Model Deployment on Embedded FPGAs for Resilient Combined Sewer Overflow Management"**  
📌 Accepted at **11th IEEE International Smart Cities Conference (2025)**  
📄 Preprint at [link](https://arxiv.org/abs/2508.13905)

If you use this repository, please consider citing our paper:
```bibtex
@misc{ling2025csoedge,
  title     = {Automating Versatile Time-Series Analysis with Tiny Transformers on Embedded FPGAs},
  author    = {Tianheng Ling, Vipin Singh, Chao Qian, Felix Biessmann and Gregor Schiele},
  year      = {2025},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2508.13905}
}
```
---

## Installation
#### 1. Clone and Set Up
```
git clone https://github.com/tianheng-ling/EdgeOverflowForecast.git
cd EdgeOverflowForecast
```
#### 2. Create and Activate Virtual Environment
```
python -m venv venv --python=python3.11
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

#### 3. Install Dependencies
```
pip install -r requirements.txt
```
---

## Dataset
For carrying out the experiments and training our models we disposed of a real-world dataset of a Combined Sewer System in the city of Duisburg, Germany, provided by the Wirtschaftsbetriebe Duisburg (WBD). The full dataset cannot be made publicly available, because of information on critical infrastructure. For further details on the dataset, please refer to the paper or contact us.

---
### Supported FPGA Platforms
This framework targets low-power embedded FPGAs, making it ideal for on-device edge AI deployment:

| FPGA Platform	Model  | Used in Papers | Frequency | Resource Budges               |
| -------------------- | -------------- | --------- | ----------------------------- |
| AMD Spartan-7 Series | XC7S15         | 100 MHz   | 8,000 LUTs, 10 BRAMs, 20 DSPs |

---

## Usage
All runnable scripts are organized in the **`scripts/`** folder for convenience:
```
scripts/
├── hardware/                     # Hardware synthesis, power & resource estimation
│   ├── hw_analysis_pipeline.sh   # Main pipeline for FPGA synthesis + analysis
│   ├── power_estimation.tcl      # Vivado power estimation script
│   ├── resource_estimation.tcl   # Vivado resource estimation script
│   └── selected_timestamp.txt    # Selected files for execute hardware synthesis
│
├── optuna/            # Hardware-aware optimizationo search
│   ├── amd/           # Configs for AMD FPGA platforms
│   ├── fp32_train.sh  # Run FP32 model search
│   └── quant_train.sh # Run quantized model search
│
└── software/      # Pure software training and evaluation
├── fp32/          # FP32 model training/evaluation scripts
└── quant/         # Quantized model QAT/evaluation scripts
```
You can **run scripts directly** from their folders.  
For example:
```bash
# Train FP32 model
bash scripts/software/fp32_train.sh

# Model QAT
bash scripts/software/quant_train.sh

# Run full FPGA synthesis and analysis after model QAT
bash scripts/hardware/hw_analysis_pipeline.sh

# Train FP32 model with Optuna search
bash scripts/optuna/fp32_train.sh

# Model QAT with bi-objective Optuna search
bash scripts/optuna/quant_train.sh
```

### Related Repositories
This project is part of a broader family of FPGA-optimized time-series models. You may also be interested in:

- **OnDeviceSoftSensorMLPs** → [GitHub Repository](https://github.com/tianheng-ling/OnDeviceSoftSensorMLP)  
- **TinyTransformer4TS** → [GitHub Repository](https://github.com/tianheng-ling/TinyTransformer4TS)  
- **OnDeviceRunningGaitRecognition** → [GitHub Repository](https://github.com/tianheng-ling/StrikeWatch)
- **OnDevice1D-(Sep)CNN** → coming soon
---

### Related Publications

This work builds upon our previous research, which provide the foundational methodologies and evaluation frameworks that inform the current implementation.

**📄 Previous Publications:**

1. [Data-driven Modeling of Combined Sewer Systems for Urban Sustainability: An Empirical Evaluation](https://www.hiig.de/wp-content/uploads/2024/09/Singh2024-SewerSystems.pdf)
   **Authors**: Vipin Singh, Tianheng Ling, Teodor Chiaburu, Felix Biessmann*  
   **Published in**: 47th German Conference on AI (2nd Workshop on Public Interest AI) 2025 

2. [Evaluating Time Series Models for Urban Wastewater Management: Predictive Performance, Model Complexity and Resilience](https://arxiv.org/abs/2504.17461)
   **Authors**: Vipin Singh, Tianheng Ling, Teodor Chiaburu, Felix Biessmann*  
   **Published in**: 10th IEEE International Conference on Smart and Sustainable Technologies (SpliTech) 2025
    **Code**: [GitHub Repository](https://github.com/calgo-lab/resilient-timeseries-evaluation)

---
###  Acknowledgement
This work is supported by the German Federal Ministry for Economic Affairs and Climate Action under the RIWWER project (01MD22007C, 01MD22007H). 


---

###  Contact
For questions or feedback, please feel free to open an issue or contact us at tianheng.ling@uni-due.de.
