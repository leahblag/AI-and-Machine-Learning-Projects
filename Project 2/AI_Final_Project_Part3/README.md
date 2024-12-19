# Fashion MNIST CNN Hyperparameter Tuning Project

## Project Overview
This project explores hyperparameter tuning for a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset. The implementation systematically evaluates the impact of different hyperparameters on model performance.

## Project Structure
```
AI_Final_Project_Part3/
├── code/
│   └── hyperparameter_exploration.py
├── models/
│   ├── model_filters_*.pt
│   ├── model_batchsize_*.pt
│   └── model_epochs_*.pt
├── figures/
│   └── hyperparameter_analysis.png
└── writeup/
    └── AI_Project_Writeup.pdf
```

## Results Summary
Key findings from hyperparameter exploration:
- Filter counts (8, 16, 32): Best performance with 32 filters (84.42%)
- Batch sizes (32, 64, 128): Optimal at 128 (86.96%)
- Training epochs (3, 5, 7): Peak performance at 5 epochs (87.39%)

## Environment Setup
Required packages:
```bash
pip install torch torchvision matplotlib numpy
```

## Running the Code
1. Ensure all dependencies are installed
2. Run the hyperparameter exploration script:
```bash
python code/hyperparameter_exploration.py
```

## Author
[Leah Blagbrough]
[Course Name: AI: Machine Learning Foundations]
[Date: December 18, 2024]