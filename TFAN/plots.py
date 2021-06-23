import tensorflow as tf
import pandas as pd
import numpy as np

from plotly import subplots
import plotly.graph_objects as go


def featurewise_attention(att, x, y_true, y_pred, feature=None,
                          show_target=False, start=None, end=None):
    """


    Parameters
    ----------
    att : scaled values norm of shape == [batch, num_heads, features,
                                          features]
    x : dataframe of features of shape == [batch, features]
    y_true : Series of target of shape == [batch, ]
    y_pred : Series of model predicitons of shape == [batch,]
    feature : str feature name
    show_target : Bool specifying whether to show the target with model
                  predictions as second subplot
    start : int specifying start index for plot
    end : int specifying end index for plot.

    Returns
    -------
    fig : plotly Figure of feature with respective attention and potentially
          with target
    """
    if start is None:
        start = 0
    if end is None:
        end = int(len(x))
    if feature is None:
        feat = x[x.columns[0]][start:end]
    else:
        feat = x[feature][start:end]
    # average attention
    features = att.shape[-1]
    att = tf.reduce_sum(att[start:end], axis=2)/float(features)
    num_heads = att.shape[1]
    att = tf.reduce_sum(att, axis=1)/float(num_heads)
    # scale between 0 and 1
    att_min = tf.expand_dims(tf.math.reduce_min(att, axis=1), 1)
    att_max = tf.expand_dims(tf.math.reduce_max(att, axis=1), 1)
    att = (att-att_min)/(att_max-att_min)
    att = tf.nn.softmax(att, axis=1)
    att = att.numpy()
    att = pd.DataFrame(data=att, columns=x.columns)

    if show_target is True:
        rows = 2
        subplot_titles = ['Attention of Feature: {}'.format(feature),
                          'Model Predictions and Target']
    else:
        rows = 1
        subplot_titles = ['Attention of Feature: {}'.format(feature)]

    fig = subplots.make_subplots(rows=rows, cols=1, print_grid=False,
                                 vertical_spacing=0.2,
                                 subplot_titles=subplot_titles)
    # feature
    feat_trace = go.Scatter(
                    x=np.arange(start, end),
                    y=feat,
                    mode="lines",
                    name=feature,
                    marker=dict(color='darkslategray'))
    # attention
    att_trace = go.Scatter(
                    x=np.arange(start, end),
                    y=att[feature],
                    mode="lines",
                    name="attention",
                    marker=dict(color='#d62728'))
    # predictions
    pred_trace = go.Scatter(
                    x=np.arange(start, end),
                    y=y_pred[start:end],
                    mode="lines",
                    name="predictions",
                    marker=dict(color='red'))
    # true values
    true_trace = go.Scatter(
                    x=np.arange(start, end),
                    y=y_true[start:end],
                    mode="lines",
                    name="target",
                    marker=dict(color='blue'))
    # append traces
    fig.append_trace(feat_trace, 1, 1)
    fig.append_trace(att_trace, 1, 1)
    if show_target:
        fig.append_trace(pred_trace, 2, 1)
        fig.append_trace(true_trace, 2, 1)
    # update y-axes
    fig['data'][1].update(yaxis="y3")  # ||att||
    if show_target:
        # fig['data'][2].update(yaxis="y2") # true
        fig['layout']["xaxis2"].update(title='Sample')
        fig['layout']["yaxis2"].update(title=y_true.name)
    else:
        fig['layout']["xaxis"].update(title='Sample')
    # overlaying axis
    fig['layout']["yaxis"].update(title=feature)
    fig['layout']["yaxis3"] = dict(
                     overlaying="y",
                     anchor="x",
                     side='right',
                     showgrid=False,
                     title='attention')
    # figure layout

    fig.update_layout(title_text="Attention of Feature: {} - t: {}-{}".format(
                                                                     feature,
                                                                     start,
                                                                     end),
                      plot_bgcolor='white', legend=dict(
                                                        yanchor="middle",
                                                        y=1.1,
                                                        xanchor="center",
                                                        x=0.8))
    return fig


def attention_distribution(att, name_of_features=[],
                           start=None, end=None):
    """


    Parameters
    ----------
    att : scaled values norm of shape == [batch, num_heads, features,
                                          features]
    name_of_features : list containing names of features
    start : int specifying start index for plot
    end : int specifying end index for plot

    Returns
    -------
    fig : plotly Figure of attention distribution across features
    df_att: Dataframe of attention distribution
    """
    if start is None:
        start = 0
    if end is None:
        end = int(att.shape[0])
    if not name_of_features:
        name_of_features = [str(num) for num in range(att.shape[-1])]

    att = att[start:end]
    # average attention
    features = att.shape[-1]
    att = tf.reduce_sum(att[start:end], axis=2)/float(features)
    num_heads = att.shape[1]
    att = tf.reduce_sum(att, axis=1)/float(num_heads)
    # scale between 0 and 1
    att_min = tf.expand_dims(tf.math.reduce_min(att, axis=1), 1)
    att_max = tf.expand_dims(tf.math.reduce_max(att, axis=1), 1)
    att = (att-att_min)/(att_max-att_min)
    att = tf.nn.softmax(att, axis=1)
    trace = go.Heatmap(z=att.numpy(),
                       x=name_of_features,
                       y=np.arange(start, end),
                       colorscale="Blues",
                       colorbar=dict(title='Attention'))
    fig = go.Figure(
        data=trace, layout=dict(
            title='Attention Distribution - start: {} - end: {}'.format(start,
                                                                        end),
            yaxis=dict(title="Samples"),
            xaxis=dict(title="Features")))
    df_att = pd.DataFrame(data=att.numpy(), columns=name_of_features)

    return fig, df_att
