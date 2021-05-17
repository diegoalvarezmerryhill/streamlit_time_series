import pywt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import streamlit as st
import statsmodels.api as sm

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

from scipy.fftpack import fft

class TimeSeries:

    def __init__(self, df, price_type, ticker):

        self.df = df
        self.price_type = price_type
        
        if self.price_type == "Adjusted Close":
            self.price_type = 'Adj Close'
            
        self.ticker = ticker

        self.df['ticker'] = self.ticker
        self.df['return'] = df[self.price_type].pct_change()
        self.df['range'] = (self.df['High'] / self.df['Low']) - 1
        self.df = self.df.dropna()
        
    def get_regimes(self):

        #splitting the data
        
        company_name = yf.Ticker(self.df['ticker'][0]).info['shortName']
        
        #splitting it up 80%
        msk = np.random.rand(len(self.df)) < 0.8
        
        train = self.df[msk]
        test = self.df[~msk]
        
        #3 discrete hidden satates: low volatility, neutral volatility, and high volatility
        X_train = train[['return', 'range', self.price_type]]
        X_test = test[['return', 'range', self.price_type]]
        
        #creating the model
        model = GaussianMixture(n_components = 3, covariance_type = 'full', n_init = 100, random_state = 7).fit(X_train)
        
        #predict the optimal sequence of interal hidden state
        hidden_states = model.predict(X_test)
        
        #what this is doing is that if finds that parameters that define each state
        for i in range(model.n_components):
            
            print("{0}th hidden state".format(i))
            print("mean = ", model.means_[i])
            print("var = ", np.diag(model.covariances_[i]))
            print()
            
        sns.set(font_scale = 1.25)
        
        style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3, 
                    'font.family': u'courier prime code', 'legend.frameon': True}
        
        sns.set_style('white', style_kwds)
        
        fig, axs = plt.subplots(model.n_components, sharex = True, sharey = True, figsize = (12,9)) 
        colors = cm.rainbow(np.linspace(0, 1, model.n_components))
        
        for i, (ax, color) in enumerate(zip(axs, colors)):
            # Use fancy indexing to plot data in each state.
            mask = hidden_states == i

            ax.plot_date(X_test.index.values[mask],
                        X_test[self.price_type].values[mask],
                        ".-", c=color)
            ax.set_title("{0}th hidden state".format(i), fontsize=16, fontweight='demi')
        
            # Format the ticks.
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            sns.despine(offset=10)
            
        plt.tight_layout()
        
        sns.set(font_scale=1.5)
        states = (pd.DataFrame(hidden_states, columns=['states'], index=X_test.index)
                .join(X_test, how='inner')
                .reset_index(drop=False)
                .rename(columns={'index':'Date'}))
        states.head()
        
        #suppressing warnings because of some issues with the font package
        #in general, would not rec turning off warnings.
        import warnings
        warnings.filterwarnings("ignore")
        
        sns.set_style('white', style_kwds)
        order = [0, 1, 2]
        fg = sns.FacetGrid(data=states, hue='states', hue_order=order,
                        palette=colors, aspect=1.31, height=12)
        
        sns.despine(offset=10)
        st.subheader("Historical {} {} regimes".format(company_name, self.price_type))
        st.pyplot(fg.map(plt.scatter, 'Date', self.price_type, alpha=0.8).add_legend())
        
        st.write(fig)
        
    def smoothed_probability(self, frequency):
        
        series = self.df[self.price_type].pct_change().to_frame().dropna()
        company_name = yf.Ticker(self.df['ticker'][0]).info['shortName']
        
        st.subheader("{}".format(company_name) + " {} returns".format(frequency))
        st.line_chart(series)
        
        mod_kns = sm.tsa.MarkovRegression(series, k_regimes=3, trend = 'nc', switching_variance = 'True')
        res_kns = mod_kns.fit()
        
        st.subheader("Smoothed probability of a low-variance regime for {}".format(company_name) + " {} returns".format(frequency))
        st.line_chart(res_kns.smoothed_marginal_probabilities[0])
        
        st.subheader("Smoothed probability of a medium-variance regime for {}".format(company_name) + " {} returns".format(frequency))
        st.line_chart(res_kns.smoothed_marginal_probabilities[1])
        
        st.subheader("Smoothed probability of a high-variance regime for {}".format(company_name) + " {} returns".format(frequency))
        st.line_chart(res_kns.smoothed_marginal_probabilities[2])
     
    def get_ave_values(self, xvalues, yvalues, n = 5):
    
        signal_length = len(xvalues)
    
        if signal_length % n == 0:
            padding_length = 0
    
        else:
            padding_length = n - signal_length//n % n
    
        xarr = np.array(xvalues)
        yarr = np.array(yvalues)
        xarr.resize(signal_length//n, n)
        yarr.resize(signal_length//n, n)
        xarr_reshaped = xarr.reshape((-1,n))
        yarr_reshaped = yarr.reshape((-1,n))
        x_ave = xarr_reshaped[:,0]
        y_ave = np.nanmean(yarr_reshaped, axis=1)
    
        return x_ave, y_ave
    
    def plot_signal_plus_average(self, ax, time, signal, average_over = 5):
    
        time_ave, signal_ave = self.get_ave_values(time, signal, average_over)
        ax.plot(time, signal, label='signal')
        ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
        ax.set_xlim([time[0], time[-1]])
        ax.set_ylabel('Amplitude', fontsize=16)
        ax.set_title('Signal + Time Average', fontsize=16)
        ax.legend(loc='upper right', fontsize=10)
    
    def get_fft_values(self, y_values, T, N, f_s):
    
        N2 = 2 ** (int(np.log2(N)) + 1) # round up to next highest power of 2
        f_values = np.linspace(0.0, 1.0/(2.0*T), N2//2)
        fft_values_ = fft(y_values)
        fft_values = 2.0/N2 * np.abs(fft_values_[0:N2//2])
        return f_values, fft_values
    
    def plot_wavelet(self, ax, time, signal, scales, waveletname = 'cmor', 
                     cmap = plt.cm.seismic, title = '', ylabel = '', xlabel = ''):
        
        dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
        power = (abs(coefficients)) ** 2
        period = 1. / frequencies
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        contourlevels = np.log2(levels)
        
        im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
        
        ax.set_title(title, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)
        
        yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], -1)
        
        return yticks, ylim
    
    def cwt(self):
        
        series = self.df[self.price_type].pct_change().to_frame().dropna()
        
        mod_kns = sm.tsa.MarkovRegression(series, k_regimes=3, trend = 'nc', switching_variance = 'True')
        res_kns = mod_kns.fit()
        
        #signal 1
        signal1 = res_kns.smoothed_marginal_probabilities[0]
        N = signal1.shape[0]
        t0=0
        dt=0.25
        
        time = np.arange(0, N) * dt + t0
        
        scales = np.arange(1, 128)
        title = 'Low variance probability Continuous Wavelet Transform (Power Spectrum)'
        
        ylabel = 'Period (years)'
        xlabel = 'Time'
        
        fig1 = plt.figure(figsize=(12,12))
        spec = gridspec.GridSpec(ncols=6, nrows=6)
        top_ax = fig1.add_subplot(spec[0, 0:5])
        bottom_left_ax = fig1.add_subplot(spec[1:, 0:5])
        self.plot_signal_plus_average(top_ax, time, signal1, average_over = 3)
        yticks, ylim = self.plot_wavelet(bottom_left_ax, time, signal1, scales, xlabel=xlabel, ylabel=ylabel, title=title)
        plt.tight_layout()
        
        #signal 2
        signal2 = res_kns.smoothed_marginal_probabilities[1]
        N = signal2.shape[0]
        t0=0
        dt=0.25
        
        time = np.arange(0, N) * dt + t0
        
        scales = np.arange(1, 128)
        title = 'Medium variance probability Continuous Wavelet Transform (Power Spectrum)'
        
        ylabel = 'Period (years)'
        xlabel = 'Time'
        
        fig2 = plt.figure(figsize=(12,12))
        spec = gridspec.GridSpec(ncols=6, nrows=6)
        top_ax = fig2.add_subplot(spec[0, 0:5])
        bottom_left_ax = fig2.add_subplot(spec[1:, 0:5])
        self.plot_signal_plus_average(top_ax, time, signal2, average_over = 3)
        yticks, ylim = self.plot_wavelet(bottom_left_ax, time, signal2, scales, xlabel=xlabel, ylabel=ylabel, title=title)
        plt.tight_layout()     
        
        #signal3
        signal3 = res_kns.smoothed_marginal_probabilities[2]
        N = signal2.shape[0]
        t0=0
        dt=0.25
        
        time = np.arange(0, N) * dt + t0
        
        scales = np.arange(1, 128)
        title = 'High variance probability Continuous Wavelet Transform (Power Spectrum)'
        
        ylabel = 'Period (years)'
        xlabel = 'Time'
        
        fig3 = plt.figure(figsize=(12,12))
        spec = gridspec.GridSpec(ncols=6, nrows=6)
        top_ax = fig3.add_subplot(spec[0, 0:5])
        bottom_left_ax = fig3.add_subplot(spec[1:, 0:5])
        self.plot_signal_plus_average(top_ax, time, signal3, average_over = 3)
        yticks, ylim = self.plot_wavelet(bottom_left_ax, time, signal3, scales, xlabel=xlabel, ylabel=ylabel, title=title)
        plt.tight_layout()            
        
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)
        
        
        #####################################################################################################################
        plot_marginal = st.radio("select plot to see smoothed probability of each variance regime", ("start", "plot"))
        
        if plot_marginal == "plot":
            
            st.subheader("{} daily returns".format(self.df['ticker'][0]))
            st.line_chart(series)
            
            st.subheader("Smoothed probability of a low-variance regime for {} returns".format(self.df['ticker'][0]))
            st.line_chart(res_kns.smoothed_marginal_probabilities[0])
            
            st.subheader("Smoothed probability of a medium-variance regime for {} returns".format(self.df['ticker'][0]))
            st.line_chart(res_kns.smoothed_marginal_probabilities[1])
            
            st.subheader("Smoothed probability of a high-variance regime for {} returns".format(self.df['ticker'][0]))
            st.line_chart(res_kns.smoothed_marginal_probabilities[2])
    
    def get_all(self):
        
        output = self.get_regimes()

        series = self.df[self.price_type].pct_change().to_frame().dropna()
        
        mod_kns = sm.tsa.MarkovRegression(series, k_regimes=3, trend = 'nc', switching_variance = 'True')
        res_kns = mod_kns.fit()
        
        st.subheader("{} daily returns".format(self.df['ticker'][0]))
        st.line_chart(series)
        
        st.subheader("Smoothed probability of a low-variance regime for {} returns".format(self.df['ticker'][0]))
        st.line_chart(res_kns.smoothed_marginal_probabilities[0])
        
        st.subheader("Smoothed probability of a medium-variance regime for {} returns".format(self.df['ticker'][0]))
        st.line_chart(res_kns.smoothed_marginal_probabilities[1])
        
        st.subheader("Smoothed probability of a high-variance regime for {} returns".format(self.df['ticker'][0]))
        st.line_chart(res_kns.smoothed_marginal_probabilities[2])        
        
        #signal 1
        signal1 = res_kns.smoothed_marginal_probabilities[0]
        N = signal1.shape[0]
        t0=0
        dt=0.25
        
        time = np.arange(0, N) * dt + t0
        
        scales = np.arange(1, 128)
        title = 'Low variance probability Continuous Wavelet Transform (Power Spectrum)'
        
        ylabel = 'Period (years)'
        xlabel = 'Time'
        
        fig1 = plt.figure(figsize=(12,12))
        spec = gridspec.GridSpec(ncols=6, nrows=6)
        top_ax = fig1.add_subplot(spec[0, 0:5])
        bottom_left_ax = fig1.add_subplot(spec[1:, 0:5])
        self.plot_signal_plus_average(top_ax, time, signal1, average_over = 3)
        yticks, ylim = self.plot_wavelet(bottom_left_ax, time, signal1, scales, xlabel=xlabel, ylabel=ylabel, title=title)
        plt.tight_layout()
        
        #signal 2
        signal2 = res_kns.smoothed_marginal_probabilities[1]
        N = signal2.shape[0]
        t0=0
        dt=0.25
        
        time = np.arange(0, N) * dt + t0
        
        scales = np.arange(1, 128)
        title = 'Medium variance probability Continuous Wavelet Transform (Power Spectrum)'
        
        ylabel = 'Period (years)'
        xlabel = 'Time'
        
        fig2 = plt.figure(figsize=(12,12))
        spec = gridspec.GridSpec(ncols=6, nrows=6)
        top_ax = fig2.add_subplot(spec[0, 0:5])
        bottom_left_ax = fig2.add_subplot(spec[1:, 0:5])
        self.plot_signal_plus_average(top_ax, time, signal2, average_over = 3)
        yticks, ylim = self.plot_wavelet(bottom_left_ax, time, signal2, scales, xlabel=xlabel, ylabel=ylabel, title=title)
        plt.tight_layout()     
        
        #signal3
        signal3 = res_kns.smoothed_marginal_probabilities[2]
        N = signal2.shape[0]
        t0=0
        dt=0.25
        
        time = np.arange(0, N) * dt + t0
        
        scales = np.arange(1, 128)
        title = 'High variance probability Continuous Wavelet Transform (Power Spectrum)'
        
        ylabel = 'Period (years)'
        xlabel = 'Time'
        
        fig3 = plt.figure(figsize=(12,12))
        spec = gridspec.GridSpec(ncols=6, nrows=6)
        top_ax = fig3.add_subplot(spec[0, 0:5])
        bottom_left_ax = fig3.add_subplot(spec[1:, 0:5])
        self.plot_signal_plus_average(top_ax, time, signal3, average_over = 3)
        yticks, ylim = self.plot_wavelet(bottom_left_ax, time, signal3, scales, xlabel=xlabel, ylabel=ylabel, title=title)
        plt.tight_layout()            
        
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)
