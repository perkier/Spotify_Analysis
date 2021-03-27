# Data Science Libraries
import pandas as pd
import numpy as np
from scipy import stats

# Machine Learning Libraries
from sklearn.preprocessing import LabelEncoder

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Utilities
import re
from collections import Counter

# NLTK -  Library to Play with Natural Language
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class Clean_Data(object):

    def __init__(self, df):

        self.df = df

        self.reduce_mem_usage()


    def memory_total_reduction(self):

        print('Mem. usage decreased from {:5.2f}Mb  to {:5.2f}Mb ({:.1f}% reduction)'.format(self.start_mem, self.end_mem, 100 * (self.start_mem - self.end_mem) / self.start_mem))


    def reduce_mem_usage(self, verbose=True):

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        self.start_mem = self.df.memory_usage().sum() / 1024 ** 2

        for col in self.df.columns:

            col_type = self.df[col].dtypes

            if col_type in numerics:

                c_min = self.df[col].min()
                c_max = self.df[col].max()

                if str(col_type)[:3] == 'int':

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)

                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)

                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)

                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df[col] = self.df[col].astype(np.int64)

                else:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df[col] = self.df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)

                    else:
                        self.df[col] = self.df[col].astype(np.float64)

            elif col_type == "object":
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore', downcast='float')

        self.end_mem = self.df.memory_usage().sum() / 1024 ** 2

        if verbose:
            print('Mem. usage decreased from {:5.2f}Mb  to {:5.2f}Mb ({:.1f}% reduction)'.format(self.start_mem, self.end_mem, 100 * (self.start_mem - self.end_mem) / self.start_mem))


    def remove_characters(self, verbose=True):

        def operation(text, stem=False):

            stop_words = stopwords.words("english")
            stemmer = SnowballStemmer("english")
            TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

            text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
            tokens = []

            for token in text.split():
                if token not in stop_words:
                    if stem:
                        tokens.append(stemmer.stem(token))
                    else:
                        tokens.append(token)

            return " ".join(tokens)

        operation_start_mem = self.end_mem

        self.df.text = self.df.text.apply(lambda x: operation(x))

        self.end_mem = self.df.memory_usage().sum() / 1024 ** 2

        if verbose:
            print('Mem. usage decreased from {:5.2f}Mb  to {:5.2f}Mb ({:.1f}% reduction)'.format(operation_start_mem, self.end_mem, 100 * (operation_start_mem - self.end_mem) / operation_start_mem))


class Pre_Processing(object):

    def __init__(self, df):

        self.df = df

    def get_missing_data(self):

        total = self.df.isnull().sum().sort_values(ascending=False)
        percent = (self.df.isnull().sum() / self.df.isnull().count() * 100).sort_values(ascending=False)

        self.missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    def plot_missing_data(self):

        self.get_missing_data()

        data_to_plot = self.df.isnull().astype(int)
        columns = data_to_plot.columns

        # convert the list to a 2D NumPy array
        data_to_plot = np.array(data_to_plot).reshape((len(data_to_plot.columns), len(data_to_plot)))
        h, w = data_to_plot.shape

        fig = plt.figure(figsize=(19, 15))
        ax = plt.subplot(111)

        im = ax.matshow(data_to_plot, cmap='binary_r', vmin=0, vmax=1)

        plt.yticks(np.arange(h), columns, fontsize=14)

        ax.set_aspect(w / h)

        plt.colorbar(im, cax = fig.add_axes([0.78, 0.5, 0.03, 0.38]))

        plt.title('Missing Data', fontsize=16)

        plt.show()
        plt.clf()


    def get_correlations(self):

        self.corrs = self.df.corr()

        self.correlation = self.correlations(self.corrs)


    class correlations():

        def __init__(self, df):

            self.corr_df = df

        def print(self):
            print(self.corr_df)

        def plot(self):

            f = plt.figure(figsize=(19, 15))

            plt.matshow(self.corr_df, fignum=f.number)

            plt.xticks(range(self.corr_df.shape[1]), self.corr_df.columns, fontsize=14, rotation=45)
            plt.yticks(range(self.corr_df.shape[1]), self.corr_df.columns, fontsize=14)

            cb = plt.colorbar(cmap='jet')
            cb.ax.tick_params(labelsize=14)

            plt.title('Correlation Matrix', fontsize=16)

            plt.show()
            plt.clf()

    def pair_plt(self, pairing_df=pd.DataFrame()):

        if len(pairing_df) == 0:
            pairing_df = self.df

        sns.pairplot(pairing_df, kind="kde")

        plt.show()
        plt.clf()

    def dist_analysis(self):

        self.data_set_dist = self.data_set_distribution(self.df)

        self.df.hist(bins=50, figsize=(20, 20))
        plt.show()
        plt.clf()

        self.df.plot.kde(figsize=(20, 20))
        plt.show()
        plt.clf()

        # sns.displot(penguins, x="flipper_length_mm", kind="kde")

    def analysis_withColors(self):

        _playlistNAME = self.df.loc[:,"playlist_name"]

        fg = sns.FacetGrid(data=self.df, hue='playlist_name', hue_order=_playlistNAME, aspect=1.61)
        # fg.map(pyplot.scatter, 'playlist_name').add_legend()

        self.df.hist(bins=50, figsize=(20, 20), color=fg)

        quit()


    class data_set_distribution(object):

        def __init__(self, df):

            self.dist_df = df

        def operation(self):
            self.target_count = Counter(self.dist_df.loc[:,'target_qualit'])

        def print(self):

            self.operation()

            print(self.target_count)

        def plot(self):

            self.operation()

            plt.figure(figsize=(16, 8))
            plt.bar(self.target_count.keys(), self.target_count.values())
            plt.title("Dataset labels distribuition")

            plt.show()
            plt.clf()

    def id_outliers(self, parameter):
        # Box plot

        sns.boxplot(x=self.df[parameter])

        plt.show()
        plt.clf()

        # Z-Score

        z = np.abs(stats.zscore(self.df))

        threshold = 3

        high_z = np.where(z > threshold)
        print(high_z)

        # IQR score - InterQuartile Range

        df_o1 = self.df

        Q1 = df_o1.quantile(0.25)
        Q3 = df_o1.quantile(0.75)
        IQR = Q3 - Q1

        print(IQR)

        # The data point where we have False that means these values are valid whereas True indicates presence of an outlier
        print(df_o1 < (Q1 - 1.5 * IQR))
        print(df_o1 > (Q3 + 1.5 * IQR))

        # Removing the outliers
        df_out_0 = self.df[(z < threshold).all(axis=1)]

        df_out_1 = df_o1[~((df_o1 < (Q1 - 1.5 * IQR)).any(axis=1))]
        df_out_1 = df_out_1[~((df_o1 > (Q3 + 1.5 * IQR)).any(axis=1))]

        # boston_df_out = boston_df_o1[~((boston_df_o1 < (Q1 - 1.5 * IQR)) | (boston_df_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]

        print(df_out_1.shape)

    def label_encoding(self):

        try:

            labels = self.tok_train.target.unique().tolist()
            labels.append("NEUTRAL")

            encoder = LabelEncoder()
            encoder.fit(self.tok_train.target.tolist())

            y_train = encoder.transform(self.tok_train.target.tolist())
            y_test = encoder.transform(self.tok_test.target.tolist())

        except:
            labels = self.orig_train.target.unique().tolist()
            labels.append("NEUTRAL")

            encoder = LabelEncoder()
            encoder.fit(self.orig_train.target.tolist())

            y_train = encoder.transform(self.orig_train.target.tolist())
            y_test = encoder.transform(self.orig_train.target.tolist())

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return y_train, y_test
