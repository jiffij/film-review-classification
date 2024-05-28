# film-review-classification
This is an implementation of analysing the effect of different hyperparameter and model architecture on sentimetal analysis.


## Dataset Preparation

The IMDb dataset is a widely used dataset for sentiment
analysis, consisting of 50,000 movie reviews
labeled as either positive (1) or negative (0).
The IMDb dataset is preprocessed using spaCy tokenizer
('encorewebsm') for tokenization. Tokens are
converted into numerical representations, and the vocabulary
is limited to 25,000 tokens. The data is split
into training, validation, and test sets with a ratio
of 0.35:0.15:0.5, and batched using BucketIterator to
minimize padding and speed up training.

## Result
According to request, different Deep neural network
model are compared and evaluated based on their
performance on the IMDb dataset. Table 1 are the
test result for the different models. Beside, The training
and evaluation loss and accuracy of each model
are recorded and plotted in Appendix A for further
analysis.
From the result, it is evident that both the
One-layer Feed Forward with 500 hidden units
and Bi-LSTM models exhibited superior accuracy,
which attained the highest accuracy of 86.46% and
85.87%; whereas the Recurrent Neural Network using
SGD optimizer demonstrated the lowest accuracy
at 54.70%. The performance of the Recurrent Neural
Network utilizing Word2Vec embeddings and Adam
optimizer for 50 epochs was also notable, yielding an
accuracy of 76.77%.

## Analysis
In the following, we will compare different models and
methods and analyze the root cause of their varying
performance on the sentiment analysis task. First,
we will examine the effect of different hyperparameter,
optimization method and word embedding on
the performance of the Recurrent Neural Network
model. Then further architecture will be explored
to understand the impact of different neural network
layers. In order to maintain a controlled environment
for comparison, all experiments were conducted using
the same hyperparameter, loss function and optimizer,
if not explicitly stated. The default setting
is as follow.
1
| Model                                     | Test Loss | Test Accuracy (%) |
|-------------------------------------------|-----------|-------------------|
| Recurrent Neural Network (original)        | 0.596     | 70.22             |
| Recurrent Neural Network (SGD optimizer)   | 0.685     | 54.70             |
| Recurrent Neural Network (Adagrad optimizer)| 0.591     | 69.42             |
| Recurrent Neural Network (Adam optimizer)  | 0.601     | 69.45             |
| Recurrent Neural Network (Adam optimizer 5 Epochs) | 0.630 | 65.82             |
| Recurrent Neural Network (Adam optimizer 10 Epochs)| 0.574 | 72.61             |
| Recurrent Neural Network (Adam optimizer 20 Epochs)| 0.612 | 69.02             |
| Recurrent Neural Network (Adam optimizer 50 Epochs)| 0.644 | 64.70             |
| Recurrent Neural Network (Word2Vec learning rate 0.001)| 0.513 | 76.77             |
| Recurrent Neural Network (Word2Vec learning rate 0.0001)| 0.423 | 82.22             |
| One-layer Feed Forward (500 hidden)       | 0.332     | 86.46             |
| Two-layer Feed Forward (500, 300 hidden)  | 0.360     | 85.00             |
| Three-layer Feed Forward (500, 300, 200 hidden)| 0.365 | 84.85             |
| CNN (1, 2, 3 size feature maps)           | 0.511     | 75.17             |
| LSTM                                      | 0.394     | 84.60             |
| Bi-LSTM                                   | 0.341     | 85.87             |

Table 1: Test Results for Different Models

| Hyperparameter       | Value          |
|----------------------|----------------|
| Loss function        | BCEWithLogitsLoss |
| Optimizer            | Adam           |
| No. of Epochs        | 50             |
| Learning rate        | 0.001          |


### Optimizer
The choice of optimization algorithm can greatly influence
the convergence speed and performance of
the model by avoiding local minima.In our experiment,
SGD, Adagrad, and Adam are trained with 20
epochs.
According to the result, SGD optimizer showed a
clear gap, of about 15% accuracy difference, in performance
compared to Adagrad and Adam. This suggests
that SGD may have struggled to effectively optimize
the learning rate and gradient, due to the lack
of adaptive learning rate adjustment mechanisms. By
observing Figure 2, it is clear that the RNN with SGD
has not reach total convergence due to the low initial
learning rate and may have been stuck in a local
minimum. On the other hand, the models utilizing
Adagrad and Adam optimizers demonstrated higher
convergence rate, as they adaptively adjust the learning
rate at each iteration based on the gradient magnitude
with formulation $\frac{\eta}{\sqrt{G_{t,ii}+\epsilon}}$
, where $G_{t,ii}$ is the
sum of the squares of the gradients .Therefore, the
model learning rate is higher when the gradient is
too small, vice versa.
Besides, a sudden drop in training loss is observed
in Figure 4 when using the Adam optimizer, indicating
momentum helps the overcome small local minima
or plateaus. It accumulates a decaying average
of past gradients, providing enough force to push the
model out of the suboptimal regions and continue
towards the global minimum. Therefore, Adam optimizer
is likely to achieve a better result amoung these
three options.

### Training Epoch
The choice of training epoch can also impact the performance
of the model. In this experiment, the RNN
model was trained with 5, 10, 20, 50 epochs.
According to Table 1, the RNN model achieves the
best performance after 10 epochs, with the lowest validation
loss and highest accuracy equal to 0.574 and
72.61%, respectively. The performance improves as
the number of epochs increases. Beyond 10 epochs,
the performance starts to degrade, indicating a po-
2
tential overfitting to the training data. As observed
in 5 and 6, the training and validation loss both
have a decreasing trend up to around 10 epochs, after
which, as is in Figure 8, the training loss continues
to decrease while the validation loss rise significantly.
The huge gap between training and validation
loss indicates overfitting to the training data, where
the model starts to memorize the training examples
rather than learning general patterns. Also, only 35%
of data is used for training purposes, as the remaining
65% is reserved for validation and testing. The
insufficient size of the training data might also have
contributed to the overfitting issue.

### Embedding
In the previous experiments, a linear layer was used
for the embedding of the inputs. However, a pretrained
Word2Vec embedding is also explored in this
task. Word2Vec embeddings capture semantic relationships
between words, which the pre-train process
is done through skipgram training and negative
sampling. Indeed, the RNN model performs better
(76.77%) with Word2Vec as the embedding rather
than using a newly trained linear layer(70.22%).
The improvement in performance with the
Word2Vec embedding is two-fold. Word2Vec embedding
captures more nuanced semantic relationships
between words learnt from large corpora of text,
which are hardly compare to a newly learn linear
layer especially with a small number of 17,500 training
samples. Furthermore, because Word2Vec is pretrained,
it provides a better starting point than a new
linear layer, requiring only fine-tuning. This accelerates
the convergence rate and results in improved
overall performance.
According to figure 9, the training and validation
loss of the RNN with the Word2Vec embedding both
show a oscillating and slightly decreasing trend. Possible
reason for the oscillation could be the large
learning rate, causing the model to oscillate around
the optimal solution. Another potential issue is overfitting,
as the Word2Vec embedding is pretrained.
This leads to the model converging extremely fast
and potentially overfitting during continued training.
Therefore, training the RNN model with the
Word2Vec embedding and reducing the learning rate
from 0.001 to 0.0001, Figure 10 shows a sharper decreasing
trend in training and validation loss with a
higher test accuracy of 82.22%. This has proved the
previous assumption.

## Architecture
Lastly, different neural network model architectures
are evaluated with the IMDb dataset. Namely, Feedforward
Neural Network, Convolution Neural Network,
Long Short-Term Memory Neural Network,
and Bi-directional LSTM.

### FNN
Basically three model with 1 (500), 2 (500, 300), and
3 (500, 300, 200) linear layer are compared. The
FFNN with one layer achieved the best performance
with an test accuracy of 86.42%. The test accuracy
decrease proportional to the number of linear
layer, 85% for two layer, and 84.85% for three layer.
Obviously, this indicates overfitting occurs when the
model complexity increases. When comparing Figure
11 to Figure 12, 13, a oscillating pattern is more obvious
in FNN model with two and three layer compared
to the one-layer model. This has further assured the
existance of overfitting, and one layer of 500 nodes
is sufficient to capture the necessary information for
accurate classification.
While it is quite surprising for FNN to be the best
model for overall test accuracy, surpassing LSTM and
Bi-LSTM models that are good at capturing sequential
dependencies. A possible explanation is that the
FNN models average the value in embedding dimension
before feeding into any linear layer. Although
this process may lose some sequential information,
it is possible that some embedding dimension could
learn sentiment information that is relevant to the
classification task. Resulting in a higher overall accuracy
for the FNN model.

### CNN
The CNN model is implemented indirectly. The tensor,
after passing through the embedding layer, is
3
reshaped into a 4-dimensional tensor and then permuted
to a shape of [64, 1, 229, 100], making it ready
for the convolutional layers. Where three conv2d
layer with kernel size of [1/2/3,Embeddingdim] are
used to capture different n-gram features. Then max
pooling is applied to each feature map to extract the
most important n-gram dimension features. Finally,
a linear layer of 3 unit is used to classify the target.
The test accuracy of the CNN model is 75.17%,
which is not high compare to other models. This may
be due to the fact that the CNN model focuses more
on capturing local patterns and features rather than
long-range dependencies or overall semantic meaning.

### LSTM & Bi-LSTM
The test accuracy of both LSTM (84.60%) and Bi-
LSTM (85.87%) models is quite similar and significant,
with the Bi-LSTM model slightly outperforming
the LSTM model. Both LSTM and Bi-LSTM
models are designed to capture sequential dependencies.
With the design of forget gate and memory
cells, LSTM based models are able to effectively handle
both local and global feature unlike CNN which
mostly capture local feature. Basically, LSTM-based
model would perform better than RNN model due
to the memory cell that alleviate the vanishing gradient
problem while dealing with long sequence, as
it could selectively remember information base on
the context. As for the slightly better performance
of the Bi-LSTM compares to LSTM. It may be attributed
to the bidirectional nature of the Bi-LSTM
model, which allows it to capture information from
not only past, but future contexts. This is crucial
for sentiment classification task which the sentiment
of a word depends on the neighboring word, left and
right. Therefore, the convergence speed for Bi-LSTM
is also faster than LSTM as observed from Figure 15,
16, the additional layer of backward LSTM captures
more context information per epoch.
