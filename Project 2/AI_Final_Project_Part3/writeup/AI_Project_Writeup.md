# Fashion MNIST CNN Hyperparameter Analysis Project Write-Up
Student Name: Leah Blagbrough
Course: AI: Machine Learning Foundations
Date: December 18, 2024

## Summary/Abstract
This project investigates the impact of hyperparameter tuning on a Convolutional Neural Network (CNN) designed for the Fashion MNIST dataset classification task. Through systematic exploration of three key hyperparameters—number of convolutional filters, batch size, and training epochs—we achieved a peak accuracy of 87.39% through optimal hyperparameter configuration. Our findings demonstrate the significant impact of hyperparameter selection on model performance and provide empirical evidence for optimal configurations in fashion item classification tasks.

## Objective
The primary objectives of this study were to:
1. Systematically evaluate the impact of different hyperparameters on CNN performance
2. Identify optimal hyperparameter configurations for the Fashion MNIST classification task
3. Quantify the relationships between hyperparameter choices and model accuracy
4. Provide empirical evidence for hyperparameter selection in similar computer vision tasks

## Process
### Dataset and Model Architecture
We utilized the Fashion MNIST dataset, consisting of 70,000 grayscale images (60,000 training, 10,000 testing) across 10 fashion item categories. Our CNN architecture comprised:
- Input layer accepting 28×28 grayscale images
- Two convolutional layers with variable filter counts
- Max pooling layers for dimensionality reduction
- Fully connected layers (120 → 84 → 10 neurons)
- ReLU activation functions
- Adam optimizer with 0.01 learning rate

### Hyperparameter Exploration
We investigated three key hyperparameters:
1. Number of Convolutional Filters: [8, 16, 32]
2. Batch Size: [32, 64, 128]
3. Number of Training Epochs: [3, 5, 7]

Key implementation code:
```python
def train_and_evaluate(model, train_loader, test_loader, epochs=3, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training and evaluation logic
    ...
```

## Results
### Experimental Findings
![Hyperparameter Analysis](figures/hyperparameter_analysis.png)

Our experiments revealed several key findings:

1. Filter Count Impact:
   - 8 filters: 84.34% accuracy
   - 16 filters: 83.19% accuracy
   - 32 filters: 84.42% accuracy
   - Surprisingly, performance remained relatively stable across different filter counts
   - Optimal performance achieved with 32 filters

2. Batch Size Effects:
   - 32 batch size: 82.99% accuracy
   - 64 batch size: 86.34% accuracy
   - 128 batch size: 86.96% accuracy
   - Clear trend of improved performance with larger batch sizes
   - Best performance achieved with the largest batch size tested (128)

3. Training Epochs:
   - 3 epochs: 84.92% accuracy
   - 5 epochs: 87.39% accuracy (best overall performance)
   - 7 epochs: 86.42% accuracy
   - Peak performance at 5 epochs suggests optimal training duration
   - Slight performance degradation with additional epochs indicates potential overfitting

## Conclusion
Our hyperparameter exploration revealed several important insights:
1. Filter count showed less impact than expected, with all configurations performing similarly
2. Larger batch sizes consistently improved model performance
3. Five epochs provided optimal training duration, with longer training showing diminishing returns

The optimal configuration achieved 87.39% accuracy with:
- 5 training epochs
- 128 batch size
- Standard filter configuration

## Further Steps
Future work could explore:
1. Additional hyperparameters:
   - Learning rate variations
   - Different optimizer configurations
   - Dropout rates
2. Architecture modifications:
   - Deeper networks
   - Skip connections
   - Alternative pooling strategies
3. Advanced techniques:
   - Learning rate scheduling
   - Cross-validation
   - Ensemble methods

## References
1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

2. He, K., et al. (2016). Deep residual learning for image recognition. CVPR 2016.

3. PyTorch Documentation. (2024). Retrieved from https://pytorch.org/docs/stable/index.html