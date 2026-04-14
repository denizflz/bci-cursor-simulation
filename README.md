Deep learning BCI system using EEGNet to decode motor imagery EEG signals and simulate closed-loop cursor control with confidence-based reliability analysis.

## Overview

Motor imagery refers to the mental simulation of movement without actual physical execution. Even when a person only imagines moving their left or right hand, measurable changes occur in EEG signals across different brain regions.

In this project, I used these signals to train a deep learning model (EEGNet) that predicts whether the subject imagined left-hand or right-hand movement, and then mapped these predictions into cursor movement in a simulated control system.

This project was implemented in Python using:

- **PyTorch** for building and training the EEGNet model  
- **MOABB (BNCI2014_001 dataset)** for EEG data access and benchmarking  
- **NumPy / Pandas** for data processing  
- **Matplotlib** for visualization and evaluation plots  
- **Scikit-learn** for train-test splitting and evaluation metrics

## Dataset

I used the **BNCI2014_001 dataset** from the MOABB framework.

- 288 total trials  
- 22 EEG channels  
- 1001 timepoints per trial  
- Balanced classes (left vs right motor imagery)

Each sample contains a full EEG recording segment corresponding to a single motor imagery trial.

## Pipeline

The final pipeline follows this structure:

→ Raw EEG signals  
→ Train/validation split (stratified)  
→ Standardization (computed on training set only)  
→ Reshaping to (1, channels, time)  
→ EEGNet feature extraction (temporal + spatial convolutions)  
→ Fully connected classification layer  
→ Left vs right prediction  
→ Cursor movement simulation (direction + confidence-based speed)

## Results

### Final Model Performance
| Metric | Value |
|--------|------:|
| Best Validation Accuracy | ~99% |
| Cursor Control Success Rate | ~77.6% |
| Confidence-filtered Success Rate | ~100% |
| Coverage (when acting) | ~88% |

## Repository Structure

```text
bci-cursor-simulation/
│
├── data.ipynb             # main experiments and training pipeline
├── requirements.txt
├── LISCENSE
├── .gitignore
├── README.md
│
├── saved_models/
└── eegnet_best.pt         # trained EEGNet model weights
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook
```bash
jupyter notebook notebooks/data.ipynb
```

## Future Work
- Multi-subject training
- Real-time EEG integration
- More advanced models (Transformers/RNN hybrids)
- True closed-loop control systems

## Article
I wrote a full article explaining the reasoning behind the pipeline, the failed baselines, and the progression from 50% to ~99% accuracy. You can access it from [here.](https://medium.com/@denizfiliz/how-i-built-a-closed-loop-brain-controlled-cursor-simulation-ff93e05e99fd)

## License
This project is licensed under the MIT License.
