# Fibonacci
This repository contains code from a Machine Learning homework problem. This code builds a simple Recurrent Neural Network (RNN) using PyTorch to predict the next number in a sequence. The model is trained on the first eight numbers of the Fibonacci sequence and learns to output the next number by identifying the pattern. The RNN consists of one hidden layer with 10 neurons and is trained over 30,000 epochs using the Adam optimizer and Mean Squared Error loss. Following training, the model uses only the last number in the sequence to generate each new prediction, one at a time. This process is repeated to forecast five additional Fibonacci numbers, extending the sequence based on the model’s learned pattern.
## Structure
- **scripts/**: Contains the Python script. 
- **README.md**: Overview of the homeowork problem. 
- **requirements.txt**: PyTorch, Numpy. 
## Contributing
If you’d like to contribute or report any issues, please open a Pull Request or file an Issue on this repository.
## Acknowledgments
I would like to thank Dr. Rahul Makhijani for their guidance and for allowing me to share this problem. Their insights and support were invaluable in completing this assignment successfully.