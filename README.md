# Time Series Forecast with Attention
### Overview
This repository contains various Deep Learning Architectures I implemented from scratch during my master thesis. My research work dealed with attentive models applied to time-series forecast, where the goal was ultimately to increase model interpretability and at the same time reach state-of-the-art prediction results. The original dataset used within my thesis can not be shared publicly and was thus exchanged. The list below shows which architectures I implemented:

### Architectures
1) Encoder-Decoder with Attention (mainly following the architecture of: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025))
    - This architecture can be used to predict one- ore multi-step-ahead and further to analyse to which input time-steps the model attends while predicting
2) Feature-wise Attention Network (developed myself and inspired by [Causal Discovery with Attention-Based Convolutional Neural Networks](https://mdpi.com/2504-4990/1/1/19 )       and [Attention is all you need](https://arxiv.org/abs/1706.03762))
    - This model can be used to predict one-step-ahead and further to analyse to which input features the model attends at inference
3) Temporal-Feature-Wise Attention Network

### References
If the architecture implemented is inspired or follows a paper I at least briefly summarise the paper in the [wiki](https://github.com/gianmarcobesso/Time-Series_Forecast/wiki). I also included references I found useful while implementing the different architectures in the related paper summaries.

Have fun...
