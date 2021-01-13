# Encoder Decoder with Attention

## Intro
The implementation follows the paper [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) summarised in the [wiki](https://github.com/gianmarcobesso/Time-Series_Forecast/wiki/Effective-Approaches-to-Attention-based-Neural-Machine-Translation). The implementation can be used for multi-variate time-series forecast. Further, the attention weights can be used to analyse to which input time steps the seq2seq model attends while predicting the output sequence. Based on the needs the attention weights can be analysed both at specific prediction steps or over the entire test set. 

## Requirements
```
pip install -r requirements.txt
```
## Use 
Add CSV file in data directory -> adjust cfg -> train -> evaluate attention weights on test set...

## Use-cases of attention weights analysis
### Is the model attending from different time steps at inference?
The attention across the input time steps can be analysed at any given time step of the test set. Below, you see two examples of how the attention was behaving at two different time steps (for the data I used during my thesis). 

### Size of input window size?
The decision of the size of the window used as input to a model is a critical choice for time-series forecast. This model can be used to asses this task by analysing the attention weights across the input time steps. In my case, I trained the model with [20,30,50] time steps as input features and subsequently analysed the [0.1,0.5,0.9] quantiles of the attention weights across the test data. 
