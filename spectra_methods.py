# MDGSC reduction pipeline
# by William Balmer

# spectrum wavelength "calibration" and visualization

# this script is not yet fully documented and user friendly!! use at own risk

### Imports ###

# generic
import numpy as np
import pandas as pd
# astronomy
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from astropy.io import fits
from astropy import units as u
# plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy.visualization import quantity_support
quantity_support()  # for getting units on the axes below  

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

### Wavelength "calibration" ###

# read in spectrum
# TODO: there's gotta be some better method than ds9 projections
b03cyg = np.loadtxt('b03cyg_extract.dat')

# function to start spectrum at peak of 0th order
def specalign(specarr):
    lamb = specarr[:,0]
    flux = specarr[:,1]
    zeroth = flux.argmax()
    print('zeroth order is at pixel '+str(zeroth))
    flux2 = flux[zeroth:]
    lamb2 = lamb[zeroth:]
    lamb2 = [x-zeroth for x in lamb2]
    return lamb2, flux2

# "register" spec on 0th order
lamb, flux = specalign(b03cyg)

# windows around the H lines to wavecal on
Hgamma = flux[530:650]
Hbeta = flux[590:650]

# determine the line minima
# TODO: actually fit a gaussian to the lines
H = [Hgamma.argmin(), Hbeta.argmin()+60]
H_pix = [0]+[x+530 for x in H]

# known wavelength points to fit to
Hb_true = 486.135
Hg_true = 434.0472
H_true = [0, Hg_true, Hb_true]

y = H_true
x = H_pix
fit = np.poly1d(np.polyfit(x, y, 1))

low = 380
high = 760

spec = pd.DataFrame(columns=['lambda', 'counts'])
spec['lambda'] = fit(lamb)
spec['counts'] = flux
spec = spec[spec['lambda']>low]
spec = spec[spec['lambda']<high]

spec.to_csv(path_or_buf='29cyg_wavecal.txt', sep='\t', columns=None, header=False, index=False)

### Visualization ###

# function to make a cute spectrum (this got me retweeted by the official matplotlib account)
def prettyspec(x,y,*kwargs):    
    # select how to color
    with plt.style.context('dark_background'):
        cmap = plt.get_cmap('Spectral_r')
        norm = plt.Normalize(x.min(), x.max())

        # get segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, axs = plt.subplots(1, 1, figsize=(16,8))

        lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=10)
        # Set the values used for colormapping
        lc.set_array(x)
        lc.set_linewidth(1)
        line = axs.add_collection(lc)
        axs.set_xlim(x.min(), x.max())
        axs.set_ylim(y.min(), y.max())
        
        plt.xlabel('$\lambda$ ({})'.format('nm')) 
        plt.ylabel('Counts ({})'.format('adu')) 
        axs.set_axisbelow(True)
        axs.yaxis.grid(color='gray', linestyle='dashed')
        axs.xaxis.grid(color='gray', linestyle='dashed')
    return

# plot the spectrum
prettyspec(spec['lambda'].to_numpy(), spec['counts'].to_numpy())
# annotate the locations of the Hydrogen lines
hdict = {H_true[1]:r'H$\gamma$', H_true[2]:r'H$\beta$'}
for line in H_true[1:]:
    plt.annotate(hdict[line], xy=(line,4800), xytext=(0, -100), color='white',
                 arrowprops=dict(color='white', arrowstyle='-|>'), xycoords='data', textcoords='offset points'
                 )
# save the figure
plt.suptitle('29 Cyg, A2V $\delta$ Scuti', color='white')
plt.savefig('29cyg_spec.png', dpi=300)
