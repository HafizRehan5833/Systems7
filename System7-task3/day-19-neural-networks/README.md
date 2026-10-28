# Day 5 – Neural Networks and Keras (Module C)

**Commit:** `day-19-neural-networks`

## Topics
- Perceptrons, neurons, weights, biases, activation functions
- Forward propagation, backpropagation, loss functions, gradient descent
- Optimizers: SGD and Adam
- TensorFlow/Keras workflow: build, compile, train, evaluate
- Practical task: predict student results using a neural network (in `tasks/moduleC_student_results/`)

## Practice questions
| # | Level | File | Question |
| --- | --- | --- | --- |
| 14 | Easy | `practice/q14_perceptron_and.py` | Single perceptron in NumPy that learns the AND gate |
| 15 | Medium | `practice/q15_keras_sgd_vs_adam.py` | Keras Sequential model on tabular data; loss curves for SGD vs Adam |

Setup note: `pip install -r requirements.txt` installs TensorFlow. Question 15 uses `tasks/data/students.csv`.

## Learning notes
- **Activation functions:** What do ReLU, sigmoid and softmax do, and where would you use each?
- **Learning rate, batch size, epochs:** How does each change training speed, stability and results? Describe one experiment you ran.
  - My note:

## Completed today
- [ ] Solved Q14 and Q15
- [ ] Completed the student results task (Module C)
- [ ] Wrote the two notes above
