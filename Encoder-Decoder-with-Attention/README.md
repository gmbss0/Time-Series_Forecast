# Encoder Decoder with Attention

## Intro
The implementation follows the paper [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) summarised in the [wiki](https://github.com/gianmarcobesso/Time-Series_Forecast/wiki/Effective-Approaches-to-Attention-based-Neural-Machine-Translation). The implementation can be used for multi-variate time-series forecast. The attention weights can be used to analyse to which input time steps the seq2seq model attentds while predicting the output sequence. Based on the needs the attention weights can be analysed both at specific prediction steps or over the entire test set. 

## Requirements
```
pip install -r requirements.txt
```
## Use 
Exchange CSV file in data directory, train, evaluate attention weights on test set...

