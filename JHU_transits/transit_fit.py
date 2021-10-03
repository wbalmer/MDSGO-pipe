### imports ###
# transit fitting suite
import juliet
# general data handling
import numpy as np
import pandas as pd
# BJD time correction tools
from astropy import coordinates as coord, units as u
from astropy.time import Time

# plotting
import matplotlib.pyplot as plt
# setting plotting style
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = 'cm'
plt.rcParams["font.family"] = "sans-serif"   # Fonts
from matplotlib import rc
rc('text', usetex=True)

### user inputs ###
target_name = "TOI-1608"
T_c = [2458792.459651,.0005086198]
plot_time = 2459000
baseindex = [1100,1280]
P = [2.47272155931859,.00010807506]
target_coords = ["03 23 12.1412544434", "+33 04 42.181053963"]
aperture_photometry_file = 'TOI-1608-apphot.xls'
sigma_clip_factor = 10
bin_factor = 10

### the code ###

# read in aperture photometry
apphot = pd.read_csv(aperture_photometry_file, sep='\t')

# barycentric time! 
target = coord.SkyCoord(target_coords[0], target_coords[1],
                        unit=(u.hourangle, u.deg), frame='icrs')
jhuloc = coord.EarthLocation.of_address('3400 Charles St, Baltimore, MD 21218, USA')


t,f,ferr,fwhms = np.array(apphot['J.D.-2400000']+2400000), apphot['rel_flux_T1'], apphot['rel_flux_err_T1'], apphot["FWHM_Mean"]
t = Time(t, format='jd', scale='utc', location=jhuloc) 
ltt_bary = t.light_travel_time(target) 
t = t.tdb + ltt_bary
t = t.jd

Tc = Time(T_c, format='jd', scale='utc', location=jhuloc)
Tc = Tc.jd

# baseline

base1, base2 = baseindex
baselinemed = np.median(f[base1:base2])
forig = f
f = f/baselinemed
ferr = (ferr/forig)*f

if sigma_clip_factor is not None:
    from astropy.stats import sigma_clip
    f = sigma_clip(f, sigma=sigma_clip_factor, maxiters=1)
if bin_factor is not None:
    t, f, ferr = juliet.bin_data(t, f, bin_factor)

# prepare the dictionaries for juliete

times, fluxes, fluxes_error = {},{},{},{}
times['jhu'], fluxes['jhu'], fluxes_error['jhu'] = t,f,ferr

params = ['P_p1','t0_p1','p_p1','b_p1','q1_jhu','q2_jhu','ecc_p1','omega_p1',\
              'rho', 'mdilution_jhu', 'mflux_jhu', 'sigma_w_jhu', 'GP_sigma_jhu', 'GP_rho_jhu']

# Distribution for each of the parameters:
dists = ['normal','normal','uniform','uniform','uniform','uniform','fixed','fixed',\
                 'loguniform', 'fixed', 'normal', 'loguniform', 'loguniform', 'loguniform', 'loguniform']

# Hyperparameters of the distributions (mean and standard-deviation for normal
# distributions, lower and upper limits for uniform and loguniform distributions, and
# fixed values for fixed "distributions", which assume the parameter is fixed)
hyperps = [P, Tc, [0.,1.], [0.,1.], [0., 1.], [0., 1.], 0.0, 90.,\
                   [300, 10000.], 1.0, [0.,1.0], [10, 1e3], [1e-6, 1e6], [1e-4,1e3]]
priors = {}
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

# Perform the juliet fit. Load dataset first:
dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, 
                      yerr_lc = fluxes_error, GP_regressors_lc = times, #GPlceparamfile = "hd189params.txt",
                      out_folder = target_name+"_GP")#_fwhm') 
results = dataset.fit()

# Plot the GP fit and then the detrended + fit transit light curve

# gather the data
transit_plus_GP_model = results.lc.evaluate('jhu')
transit_model = results.lc.model['jhu']['deterministic']
gp_model = results.lc.model['jhu']['GP']

# figure
fig = plt.figure(constrained_layout=True, figsize=(8,12))
axd = fig.subplot_mosaic(
    """
    A
    B
    """
)

axd['A'].errorbar(times['jhu']-plot_time, fluxes['jhu'], fluxes_error['jhu'],fmt='.',alpha=0.4)
axd['A'].plot(dataset.times_lc['jhu']-plot_time, transit_plus_GP_model, color='black',zorder=10)

axd['B'].errorbar(times['jhu']-plot_time, fluxes['jhu']-gp_model, fluxes_error['jhu'],fmt='.',alpha=0.4, zorder=0, label='detrended lightcurve')
axd['B'].plot(dataset.times_lc['jhu']-plot_time, transit_model, color='xkcd:brick',zorder=1, linewidth=3, label='transit model')
t_bin, y_bin, yerr_bin = juliet.bin_data(times['jhu']-plot_time, fluxes['jhu']-gp_model, 5)
axd['B'].errorbar(t_bin, y_bin, yerr = yerr_bin, fmt = 'o', mfc = 'white', mec = 'black', ecolor = 'black', zorder=2, label='binned data')

fig.suptitle(target_name+', MDSGO', fontsize=16)
axd['A'].set_title(r'Observations and GP model fit with \texttt{Juliete}', fontsize=12)
axd['B'].set_title('Detrended lightcurve', fontsize=12)
axd['B'].legend(loc='lower right')
axd['B'].set_xlabel('Time [BJD-{}]'.format(plot_time))
axd['A'].set_ylabel('Relative Flux')
axd['B'].set_ylabel('Relative Flux')

plt.savefig(target_name+'-GP.png', dpi=300)