# Data Science Libraries
import pandas as pd
from sklearn import preprocessing

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import radviz, lag_plot


def dist_analysis(df, parameter_1):

    # Violin plots are a combination of box and KDE plots.
    # They deliver the summary statistics with the box plot inside and shape of distribution with the KDE plot on the sides.

    df[parameter_1] = df[parameter_1].astype(float).round(decimals=2)

    df = df.sort_values(by='playlist_name', ascending=False)

    sns.violinplot(x=parameter_1, y='playlist_name', data=df, inner="quart", linewidth=1, bw=.5)

    plt.show()
    plt.clf()


def check_randomness(df, parameter):

    # Lag plots are used to check if a data set is random

    lag_plot(df.loc[:,parameter])
    plt.show()
    plt.clf()


def rad(df, distinction, parameters):

    # RadViz is a way of visualizing multi-variate data.
    # Based on a simple spring tension minimization algorithm.

    df_label = df.loc[:,distinction]

    df = df.loc[:,parameters]

    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=parameters)

    df[distinction] = df_label

    plt.figure()

    radviz(df, distinction, color=["cyan", "lime", "fuchsia", "maroon"])

    plt.show()
    plt.clf()


def correlation_plots(df, parameters):

    corr_df = df.loc[:,parameters].corr()

    f = plt.figure(figsize=(19, 15))

    plt.matshow(corr_df, fignum=f.number)

    plt.xticks(range(corr_df.shape[1]), corr_df.columns, fontsize=14, rotation=45)
    plt.yticks(range(corr_df.shape[1]), corr_df.columns, fontsize=14)

    cb = plt.colorbar(cmap='jet')
    cb.ax.tick_params(labelsize=14)

    plt.title('Correlation Matrix', fontsize=16)

    plt.show()
    plt.clf()
