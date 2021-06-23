# Temporal Featurewise Attention Network (TFAN)
I developed this architecture during my thesis to allow interpretable multi-variate time series forecast. The model allows to assess the relevance of individual features at inference. This is particularly valuable, when dealing with data, where the relevance of features strongly varies through time. Typically, a convolutional or recurrent architecture combines the information of the different features from the very first layer. Increasing depth of a network and non-linear activations quickly make an assessment of the per-feature information content unfeasible, as it would require to backtrace all computations. Thus, I chose another approach. The information of every feature is processed in parallel (no inter-feature interaction) first with causal (temporal) convolutions. Subsequently, the extracted information of the different features is combined with Multi-Head-Attention. The attention is thus responsible to determine how to combine the per-feature information (which feature is relevant?). Hence, analysing the attention allows to interpret the model. Finally, some causal convolutions lead to the predicition of the target time series.

## Architecture
The architeecture consists of: depthwise 1D convolutions, Multi-Head-Attention(MHA) and final 1D convolutions.
![image](https://user-images.githubusercontent.com/62936465/123063000-4f001180-d40d-11eb-8764-754821c195d2.png)
### Depthwise Conv1D
Unfortunately, tensorflow does not support a 1D implementation of DepthwiseConv2D. A depthwise convolution refers to no combining the individual channels of an input to a convolution (in this case the features). Thus, I had to implement this layer myself by combining different convolution layers. Hopefully, tensorflow will include this layer soon...
TFAN basically applies an individual TCN (see my short summary of the paper [here](https://github.com/gmbss0/Time-Series_Forecast/wiki/An-Empirical-Evaluation-of-Generic-Convolutional-and-Recurrent-Networks-for-Sequence-Modeling-(TCN-paper))) to every feature. I renounce on the Weight Normalization though. Feel free to experiment with different amount of residual blocks varying the depth of the network.
### MHA
The Multi-Head-Attention implementaion follows the [tensorflow code](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention). However, for my attention analysis I require the values (that are not returned by the tf implementation). Additionally, I included the regularization method [dropAttention](https://arxiv.org/abs/1907.11065).
### Final 1D Convolutions
After combining the information of the different features, a series of classic 1D convolutions is performed. This leads to the prediction of the output time series.
### Attention Analysis
The attention analysis is based on the findings of [this paper](https://arxiv.org/abs/2004.10102), summarised in the [wiki](https://github.com/gmbss0/Time-Series_Forecast/wiki/Attention-is-Not-Only-aWeight:-Analyzing-Transformers-with-Vector-Norms). The attention analysis includes the attention weights, values and trained layer weights to effectively compute a quantitative assessment of the information content of the individual features.

## Requirements
```
pip install -r requirements.txt
```
## Use 
Add CSV files in data directory -> adjust cfg -> train model in TFAN.py -> evaluate attention on test set...

## Results
The following images show results of my research work. The actual feature names are anonymised. Both plot types can be generated with the code in TFAN.py.
### Attention distribution
The attention distribution shows the attention of the model across features and time.
![image](https://user-images.githubusercontent.com/62936465/123095042-e7a68980-d42d-11eb-817e-7a70f7183a1e.png)
### Featurewise attention
![image](https://user-images.githubusercontent.com/62936465/123094501-3dc6fd00-d42d-11eb-8a7d-cb6d31059755.png)

