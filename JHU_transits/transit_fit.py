# imports
import juliet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import coordinates as coord, units as u
from astropy.time import Time

# read in aperture photometry
apphot = pd.read_csv('hd189_19-9-21_aperture_phot.xls', sep='\t')

# barycentric time! 
hd189 = coord.SkyCoord("20:00:43.7130382888", "+22:42:39.071811263",
                        unit=(u.hourangle, u.deg), frame='icrs')
jhuloc = coord.EarthLocation.of_address('3400 Charles St, Baltimore, MD 21218, USA')


t,f,ferr,fwhms = np.array(apphot['J.D.-2400000']), apphot['rel_flux_T1'], apphot['rel_flux_err_T1'], apphot["FWHM_Mean"]
t = Time(t, format='mjd', scale='utc', location=jhuloc) 
ltt_bary = t.light_travel_time(hd189) 
t = t.tdb + ltt_bary
t = t.jd

Tc = Time(59477.558, format='mjd', scale='utc', location=jhuloc)
lttc_bary = Tc.light_travel_time(hd189)
Tc = Tc.tdb + lttc_bary
Tc = Tc.jd

# baseline

baselinemed = np.median(f[1300:])
forig = f
f = f/baselinemed
ferr = (ferr/forig)*f

# prepare the dictionaries for juliete

times, fluxes, fluxes_error, fwhm = {},{},{},{}
times['jhu'], fluxes['jhu'], fluxes_error['jhu'], fwhm['jhu'] = t,f,ferr,fwhms

params = ['P_p1','t0_p1','p_p1','b_p1','q1_jhu','q2_jhu','ecc_p1','omega_p1',\
              'rho', 'mdilution_jhu', 'mflux_jhu', 'sigma_w_jhu', 'GP_sigma_jhu', 'GP_rho_jhu']

# Distribution for each of the parameters:
dists = ['normal','normal','uniform','uniform','uniform','uniform','fixed','fixed',\
                 'loguniform', 'fixed', 'normal', 'loguniform', 'loguniform', 'loguniform', 'loguniform']

# Hyperparameters of the distributions (mean and standard-deviation for normal
# distributions, lower and upper limits for uniform and loguniform distributions, and
# fixed values for fixed "distributions", which assume the parameter is fixed)
hyperps = [[2.21857567,0.00000015], [Tc,.05], [0.05,.3], [0.2,0.8], [0., 1.], [0., 1.], 0.0, 90.,\
                   [300, 10000.], 1.0, [0.,1.0], [10, 1e3], [1e-6, 1e6], [1e-4,1e3]]
priors = {}
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

# Perform the juliet fit. Load dataset first:
dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, 
                      yerr_lc = fluxes_error, GP_regressors_lc = times, #GPlceparamfile = "hd189params.txt",
                      out_folder = 'hd189_GP')#_fwhm') 
results = dataset.fit()

# Plot the GP fit and then the detrended + fit transit light curve
transit_plus_GP_model = results.lc.evaluate('jhu')
transit_model = results.lc.model['jhu']['deterministic']
gp_model = results.lc.model['jhu']['GP']

fig = plt.figure(constrained_layout=True, figsize=(8,12))
axd = fig.subplot_mosaic(
    """
    A
    B
    """
)

axd['A'].errorbar(times['jhu'], fluxes['jhu'], fluxes_error['jhu'],fmt='.',alpha=0.4)
axd['A'].plot(dataset.times_lc['jhu'], transit_plus_GP_model, color='black',zorder=10)

axd['B'].errorbar(times['jhu'], fluxes['jhu']-gp_model, fluxes_error['jhu'],fmt='.',alpha=0.4, zorder=0, label='detrended lightcurve')
axd['B'].plot(dataset.times_lc['jhu'], transit_model, color='xkcd:brick',zorder=1, linewidth=3, label='transit model')
t_bin, y_bin, yerr_bin = juliet.bin_data(times['jhu'], fluxes['jhu']-gp_model, 25)
axd['B'].errorbar(t_bin, y_bin, yerr = yerr_bin, fmt = 'o', mfc = 'white', mec = 'black', ecolor = 'black', zorder=2, label='binned data')

fig.suptitle('HD 189733 Transit 19/9/2021, MDSGO', fontsize=16)
axd['A'].set_title('Observations and simple GP model fit', fontsize=12)
axd['B'].set_title('Detrended lightcurve', fontsize=12)
axd['B'].legend(loc='lower right')

plt.savefig('hd189_19-9-21_simpleGP.png', dpi=150)