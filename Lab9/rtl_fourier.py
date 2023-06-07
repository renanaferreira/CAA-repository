#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:25:19 2023

Each element in the data series represents the number of sales of an item for a specific day.
The series contains data points for 760 days -- a little over 2 years of data. 
One year is the time of a single period in the data.

@author: dnt2
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import butter, freqs
import matplotlib.pyplot as plt

from pathlib import Path
import pickle

# Output directory to pickle the figures to
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

#-------------
# Time series
#-------------
data_xlsx = "dat_rtl_upc1.csv"

data = pd.read_csv(data_xlsx, header=0, parse_dates=[0])

Fs = 365  # Number of samples in a period of one year
nDays = Fs

y1 = data[:nDays]
y2 = data[nDays:2*nDays]

fig, ax = plt.subplots()
ax.plot(range(nDays), y1.sales)
ax.plot(range(nDays), y2.sales)
ax.plot(range(nDays), y1.discount, 'r')
ax.plot(range(nDays), y2.discount, 'g')
# ax.margins(0, 0.1)
ax.grid('on')
sales_ylim = ax.get_ylim()

ax.set_ylabel("Number of sold items")
ax.set_xlabel('Day')
y1_legend = "Y1: {} to {}".format(data.date.iloc[0].date(), data.date.iloc[nDays-1].date())
y2_legend = "Y2: {} to {}".format(data.date.iloc[nDays].date(), data.date.iloc[2*nDays-1].date())
ax.legend([y1_legend, y2_legend, "Y1 discounts", "Y2 discounts"])

ax.set_title('Two years of retail sales - time domain')
fig_fname = Path(fig_dir, ax.get_title().replace(" ", "_").replace('.','p')).with_suffix(".fig.pickle")
with open(fig_fname, 'wb') as file:
    pickle.dump(fig, file)
fig.set_layout_engine('tight')
fig.savefig(fig_fname.with_suffix('.png'))
  
# with open(fig_fname, 'rb') as file:
#     figx = pickle.load(file)   
# figx.show()

# --------------------
#  Fourier Transform
# --------------------

# Bands of interest (repetitions per year)
Fc = ((0.001,3), (51,53), (103, 105), (155, 157))

# Remove the DC component from the time series
sales1 = y1.sales - y1.sales.mean()
sales2 = y2.sales - y2.sales.mean()

# rfft() of the pnadas.Series caused an error, therefore we convert them to a list
sales1f = rfft(list(sales1))
sales2f = rfft(list(sales2))
xf = rfftfreq(nDays, 1/Fs)

fig, ax = plt.subplots()
ax.plot(xf, abs(sales1f))
ax.plot(xf, abs(sales2f))
ax.grid('on')

# for Fci in Fc:
#     ax.axvline(Fci[0], c='grey', ls='--', lw=1) # low cutoff frequency
#     ax.axvline(Fci[1], c='grey', ls='--', lw=1) # high cutoff frequency
salesf_ylim = ax.get_ylim()

ax.set_xlabel('Frequency (repetitions per year)')
ax.set_ylabel('Strength of Frequency Component')
ax.legend([y1_legend, y2_legend])

ax.set_title('Two years of retail sales - frequency domain')
fig_fname = Path(fig_dir, ax.get_title().replace(" ", "_").replace('.','p')).with_suffix(".fig.pickle")
with open(fig_fname, 'wb') as file:
    pickle.dump(fig, file)
fig.set_layout_engine('tight')
fig.savefig(fig_fname.with_suffix('.png'))
    
# #----------------------------------------------------------------------------
# #  Sanity check with irfft() 
# #----------------------------------------------------------------------------

# sales1i = irfft(sales1f, nDays)
# sales2i = irfft(sales2f, nDays)
 
# fig, ax = plt.subplots()

# # subtract the min to return to the original y position of the signal
# # and because we cannot have negative sales
# ax.plot(range(nDays), sales1i - min(sales1i)) 
# ax.plot(range(nDays), sales2i - min(sales2i))
# ax.plot(range(nDays), y1.discount, 'r')
# ax.plot(range(nDays), y2.discount, 'g')
# ax.grid('on')
# ax.set_ylim(sales_ylim)

# ax.set_ylabel("Number of sold items")
# ax.set_xlabel('Day')
# ax.legend([y1_legend, y2_legend, 'Y1 discounts', "Y2 discounts"])

# ax.set_title('Two years of retail sales - time domain - after IRFFT for sanity check')
# fig_fname = Path(fig_dir, ax.get_title().replace(" ", "_").replace('.','p')).with_suffix(".fig.pickle")
# with open(fig_fname, 'wb') as file:
#     pickle.dump(fig, file)
# fig.set_layout_engine('tight')
# fig.savefig(fig_fname.with_suffix('.png'))


#-------------------------------------
#  Bandpass Butterworth analog filter
#-------------------------------------
def bandpass_butter(low_high_Fc, order):
    low_high_Omegac =2*np.pi*np.array(low_high_Fc)
    
    b, a = butter(order, low_high_Omegac, btype='band', analog=True)
    
    # Frequency Response
    omega, H = freqs(b, a, worN=2*np.pi*xf)
    F = omega/2/np.pi
    
    return F, H

# Frequency responses of the bandpass filters for the above bands
# Half the number of days due to the Nyquist limit
H = np.empty((len(Fc), int(np.ceil(nDays/2))), dtype='complex')

# Year 1 and Year 2 timeseries after passing the 
# original timeseries through the bandpass filters
band_data1 = np.empty((len(Fc), nDays))
band_data2 = band_data1.copy()

for ix in range(len(Fc)):
    F, H[ix] = bandpass_butter(Fc[ix], 7)

    low_Fc = Fc[ix][0]
    high_Fc = Fc[ix][1]

    # # Plot the magnitude of the Frequency response of the filter
    # fig, ax = plt.subplots()
    # ax.plot(F, 20 * np.log10(abs(H[ix])))
    # ax.axvline(low_Fc, color='red') # low cutoff frequency
    # ax.axvline(high_Fc, color='red') # high cutoff frequency
    # # ax.margins(0, 0.1)
    # ax.grid('on')

    # ax.set_xlabel('Frequency (repetitions per year)')
    # ax.set_ylabel('Gain [dB]')

    # ax.set_title('Butterworth filter Fc1={} Fc2={}'.format(low_Fc, high_Fc))
    # fig_fname = Path(fig_dir, ax.get_title().replace(" ", "_").replace('.','p')).with_suffix(".fig.pickle")
    # with open(fig_fname, 'wb') as file:
    #     pickle.dump(fig, file)
    # fig.set_layout_engine('tight')
    # fig.savefig(fig_fname.with_suffix('.png'))

    #----------------------------------
    # Filter selected frequency bands
    #----------------------------------
    
    # Element-wise (aka Hadamard) multiplication of the Frequency domain of the sales
    # and the Frequency response of the Butterworth filter. IFFT of the result will be
    # equivalent of what convolution in the time domain would produce.
    filtered_sales1f = sales1f * H[ix]
    filtered_sales2f = sales2f * H[ix]

    fig, ax = plt.subplots()
    ax.plot(F, abs(filtered_sales1f))
    ax.plot(F, abs(filtered_sales2f))
    ax.grid('on')
    ax.set_ylim(salesf_ylim)
    
    ax.set_xlabel('Frequency (repetitions per year)')
    ax.set_ylabel('Strength of Frequency Component')
    ax.legend([y1_legend, y2_legend])
    
    ax.set_title('Frequencies after filtering - Fc1={} Fc2={}'.format(low_Fc, high_Fc))
    fig_fname = Path(fig_dir, ax.get_title().replace(" ", "_").replace('.','p')).with_suffix(".fig.pickle")
    with open(fig_fname, 'wb') as file:
        pickle.dump(fig, file)
    fig.set_layout_engine('tight')
    fig.savefig(fig_fname.with_suffix('.png'))

    #------------------------------------------------
    # Inverse Fourier to see the filtered time series
    #------------------------------------------------
    
    band_data1[ix] = irfft(filtered_sales1f, nDays)
    band_data2[ix] = irfft(filtered_sales2f, nDays)

    fig, ax = plt.subplots()
    
    # Subtract min(band_data), since we cannot have negative sales
    ax.plot(range(nDays), band_data1[ix]-min(band_data1[ix]))
    ax.plot(range(nDays), band_data2[ix]-min(band_data2[ix]))
    ax.plot(range(nDays), y1.discount, 'r')
    ax.plot(range(nDays), y2.discount, 'g')
    ax.grid('on')
    ax.set_ylim(sales_ylim)

    ax.set_ylabel("Number of sold items")
    ax.set_xlabel('Day')
    ax.legend([y1_legend, y2_legend, 'Y1 discounts', "Y2 discounts"])

    ax.set_title('Time series after filtering - Fc1={} Fc2={}'.format(low_Fc, high_Fc))
    fig_fname = Path(fig_dir, ax.get_title().replace(" ", "_").replace('.','p')).with_suffix(".fig.pickle")
    with open(fig_fname, 'wb') as file:
        pickle.dump(fig, file)
    fig.set_layout_engine('tight')
    fig.savefig(fig_fname.with_suffix('.png'))


# Plot the magnitudes of the Frequency responses of _ALL_ filters
fig, ax = plt.subplots()  
  
for Fci, Hi in zip(Fc, H):
    ax.plot(F, 20 * np.log10(abs(Hi)))
    ax.axvline(Fci[0], c='grey', ls='--', lw=1) # low cutoff frequency
    ax.axvline(Fci[1], c='grey', ls='--', lw=1) # high cutoff frequency

ax.set_xlabel('Frequency (repetitions per year)')
ax.set_ylabel('Gain [dB]')

ax.set_title('Butterworth bandpass filters')
fig_fname = Path(fig_dir, ax.get_title().replace(" ", "_").replace('.','p')).with_suffix(".fig.pickle")
with open(fig_fname, 'wb') as file:
    pickle.dump(fig, file)
fig.set_layout_engine('tight')
fig.savefig(fig_fname.with_suffix('.png'))