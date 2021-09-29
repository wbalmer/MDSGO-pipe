"""
Originally written by M. Petersen, S. Betti, and K. Ward-Duong (Spring 2019).
2020-03-31 (v2): Updated.. (KWD)
"""

# python 2/3 compatibility
from __future__ import print_function
# numerical python
import numpy as np
# file management tools
import glob
import os
# good module for timing tests
import time
# plotting stuff
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# ability to read/write fits files
from astropy.io import fits
# fancy image combination technique
from astropy.stats import sigma_clip
from astropy.stats import SigmaClip
# median absolute deviation: for photometry
from astropy.stats import mad_std
# photometric utilities
from photutils import DAOStarFinder,aperture_photometry, CircularAperture, CircularAnnulus, Background2D, MedianBackground
# periodograms
from astropy.stats import LombScargle
from regions import read_ds9, write_ds9
from astropy.wcs import WCS
import warnings
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clipped_stats
# timing
from astropy import coordinates as coord, units as u
from astropy.time import Time

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)


def construct_astrometry(hdr_wcs):
    '''
    construct_astrometry

    make the pixel to RA/Dec conversion (and back) from the header of an astrometry.net return

    inputs
    ------------------------------
    hdr_wcs  :  header with astrometry information, typically from astrometry.net

    returns
    ------------------------------
    w        :  the WCS instance

    '''

    # initialize the World Coordinate System
    w = WCS(naxis=2)

    # specify the pixel to RA/Dec conversion
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cd = np.array([[hdr_wcs['CD1_1'],hdr_wcs['CD1_2']],[hdr_wcs['CD2_1'],hdr_wcs['CD2_2']]])
    w.wcs.crpix = [hdr_wcs['CRPIX1'], hdr_wcs['CRPIX2']]
    w.wcs.crval = [hdr_wcs['CRVAL1'],hdr_wcs['CRVAL2']]
    w.wcs.cunit = [hdr_wcs['CUNIT1'],hdr_wcs['CUNIT2']]
    w.wcs.latpole = hdr_wcs['LATPOLE']
    #w.wcs.lonpole = hdr_wcs['LONPOLE']
    w.wcs.theta0 = hdr_wcs['LONPOLE']
    w.wcs.equinox = hdr_wcs['EQUINOX']

    # calculate the RA/Dec to pixel conversion
    w.wcs.fix()
    w.wcs.cdfix()
    w.wcs.set()

    # return the instance
    return w



def StarFind(imname, FWHM, nsigma, aprad, skywidth, border_scale = 2):
    '''
    StarFind*

    find all stars in a .fits image

    inputs
    ----------
    imname: name of .fits image to open.
    FWHM: fwhm of stars in field
    nsigma: number of sigma above background above which to select sources.  (~3 to 4 is a good estimate)

    outputs
    --------
    xpos: x positions of sources
    ypos: y positions of sources
    nstars: number of stars found in image
    '''

    #open image
    im,hdr=fits.getdata(imname, header=True)

    # sarah: check to see if this breaks the original version. -mp
    im = np.array(im).astype('float')

    #determine background
    bkg_sigma = mad_std(im)

    print('begin: DAOStarFinder')

    daofind = DAOStarFinder(fwhm=FWHM, threshold=nsigma*bkg_sigma, exclude_border=True)
    sources = daofind(im)

    #x and y positions
    xpos = sources['xcentroid']
    ypos = sources['ycentroid']

    # get rid of stars on border which could have NaN photometry
    diameter = border_scale * (aprad + skywidth)

    xmax = np.shape(im)[1]
    ymax = np.shape(im)[0]

    xpos2 = xpos[(xpos > diameter) & (xpos < (xmax-diameter)) & (ypos > diameter) & (ypos < (ymax -diameter))]
    ypos2 = ypos[(xpos > diameter) & (xpos < (xmax-diameter)) & (ypos > diameter) & (ypos < (ymax -diameter))]

    #number of stars found
    nstars = len(xpos2)

    print('found ' + str(nstars) + ' stars')
    return xpos2, ypos2, nstars


def makeApertures(xpos, ypos, aprad,skybuff, skywidth):
    '''
    makeApertures

    makes a master list of apertures and the annuli

    inputs
    ---------
    xpos: list - x positions of stars in image
    ypos: list - y positions of stars in image
    aprad: float - aperture radius
    skybuff: float - sky annulus inner radius
    skywidth: float - sky annulus outer radius

    outputs
    --------
    apertures: list - list of aperture positions and radius
    annulus_apertures: list - list of annuli positions and radius
    see: https://photutils.readthedocs.io/en/stable/api/photutils.CircularAperture.html#photutils.CircularAperture
    for more details

    '''

    # make the master list of apertures
    apertures = CircularAperture(np.array((xpos.astype('int'), ypos.astype('int'))).T, r=aprad)
    annulus_apertures = CircularAnnulus(np.array((xpos.astype('int'), ypos.astype('int'))).T, r_in=aprad+skybuff, r_out=aprad+skybuff+skywidth)
    apers = [apertures, annulus_apertures]

    return apertures, annulus_apertures

def apertureArea(apertures):
    ''' returns the area of the aperture'''
    return apertures.area  ### should be apertures

def backgroundArea(back_aperture):
    '''returns the area of the annuli'''
    return back_aperture.area ### should be annulus_apertures


def doPhotometry(imglist, xpos, ypos, aprad, skybuff, skywidth,timekey='MJD-OBS',verbose=1):
    '''
    doPhotomoetry*

    determine the flux for each star from aperture photometry

    inputs
    -------
    imglist: list - list of .fits images
    xpos, ypos: lists - lists of x and y positions of stars
    aprad, skybuff, skywidth: floats - aperture, sky annuli inner, sky annuli outer radii

    outputs
    -------
    Times: list - time stamps of each observation from the .fits header
    Photometry: list - aperture photometry flux values found at each xpos, ypos position

    '''

    # number of images
    nimages = len(imglist)
    print('Found {} images'.format(nimages))

    #create lists for timestamps and flux values
    Times = []
    Photometry = []

    print('making apertures')
    # make the apertures around each star
    apertures, annulus_apertures = makeApertures(xpos, ypos, aprad, skybuff, skywidth)
    annulus_masks = annulus_apertures.to_mask(method='center')

    # plot apertures on full image
    plt.figure(figsize=(6,6))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(fits.getdata(imglist[1]))
    plt.imshow(fits.getdata(imglist[1]), vmin=vmin,vmax=vmax, origin='lower')
    apertures.plot(color='white', lw=2)
    #annulus_apertures.plot(color='red', lw=2)
    plt.title('Apertures - Full Image')
    plt.show()

    # plot apertures on a subset of the image to examine the annuli
    plt.figure(figsize=(6,6))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(fits.getdata(imglist[1])[0:800,0:800])
    plt.imshow(fits.getdata(imglist[1])[0:800,0:800], vmin=vmin,vmax=vmax, origin='lower')
    apertures.plot(color='white', lw=2)
    annulus_apertures.plot(color='red', lw=2)
    plt.title('Apertures - Inset Image')
    plt.show()


    #determine area of apertures
    area_of_ap = apertureArea(apertures)

    #determine area of annuli
    area_of_background = backgroundArea(annulus_apertures)

    checknum = np.linspace(0,nimages,10).astype(int)


    #go through each image and run aperture photometry
    for ind in np.arange(nimages):

        if ((ind in checknum) & (verbose==1)):
            print('running aperture photometry on image: ', ind )

        if (verbose!=1):
            print('running aperture photometry on image: ', ind )

        #open image
        data_image, hdr = fits.getdata(imglist[ind], header=True)

        # find time stamp and append to list
        #
        # 'MJD-OBS' is specific to the WIYN 0.9m headers. I changed it
        # so that's the default, but you can pass something
        # different. -mp

        tobs = Time(hdr[timekey])
        Times.append(tobs.jd)

        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(data_image)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d) #default is 3sigma
            bkg_median.append(median_sigclip)
        bkg_median = np.array(bkg_median)

        # do photometry
        phot = aperture_photometry(data_image, apertures)
        flux0 = phot['aperture_sum'] - (bkg_median * apertures.area)

        nan = np.where((np.isnan(flux0)) | (np.isinf(flux0)))
        if len(nan[0]) > 0:
            bad_starx = xpos[nan]
            bad_stary = ypos[nan]

            print(f'image {ind} NaN flux star (x,y): {bad_starx, bad_stary}')

        #append to list
        Photometry.append(flux0)

    Photometry = np.array(Photometry)
    Times = np.array(Times)
    return Times,Photometry



def doPhotometryError(imglist,xpos, ypos,aprad, skybuff, skywidth, flux0, GAIN=1.3, manual = False, **kwargs):
    '''
    doPhotometryError

    determine error in photometry from background noise
    two options:
    - use sigma clipping and use whole background
    - manually input background box positions as kwargs

    inputs
    --------
    imglist: list - list of .fits images
    xpos, ypos: lists - lists of x and y positions of stars
    aprad, skybuff, skywidth: floats - aperture, sky annuli inner, sky annuli outer radii
    flux0: list - aperture photometry found from doPhotometry() function
    GAIN: float - average gain
    manual: boolean - switch between manually inputting box (True) or using sigma clipping (False)
                        if True -- must have kwargs
                        manual = False is default
    **kwargs
        kwargs[xboxcorner]: float - x edge of box in pixel coords
        kwargs[yboxcorner]: float - y edge of box in pixel coords
        kwargs[boxsize]: float - size of box in pixel coords


    '''

    # find number of images in list
    nimages = len(imglist)
    print('Found {} images'.format(nimages))

    #make apertures
    apertures, annulus_apertures = makeApertures(xpos, ypos, aprad, skybuff, skywidth)


    #find areas of apertures and annuli
    area_of_ap = apertureArea(apertures)

    area_of_background = backgroundArea(annulus_apertures)

    checknum = np.linspace(0,nimages,10).astype(int)


    #find error in photometry
    ePhotometry = []
    for ind in np.arange(nimages):
        #open images
        im = fits.getdata(imglist[ind])

        if ind in checknum:
            print('running error analysis on image ', ind)

        #determine variance in background
        if manual == True: #manual method -- choose back size
            skyvar = np.std(im[kwargs['xboxcorner']:kwargs['xboxcorner']+kwargs['boxsize'],kwargs['yboxcorner']:kwargs['yboxcorner']+kwargs['boxsize']])**2.
            err1 = skyvar*(area_of_ap)**2./(kwargs['boxsize']*kwargs['boxsize'])  # uncertainty in mean sky brightness
        if manual == False: #automatic method -- use sigma clipping
            filtered_data = sigma_clip(im, sigma=3)
            skyvar = np.std(filtered_data)**2.
            err1 = skyvar*(area_of_ap)**2./(np.shape(im[0])[0]*np.shape(im[1])[0])  # uncertainty in mean sky brightness

        err2 = area_of_ap * skyvar  # scatter in sky values

        err3 = flux0[ind]/GAIN # Poisson error
        if ind in checknum:
            print ('Scatter in sky values: ',err2**0.5,', uncertainty in mean sky brightness: ',err1**0.5)

        # sum souces of error in quadrature
        errtot = (err1 + err2 + err3)**0.5

        nan = np.where((np.isnan(errtot)) | (np.isinf(errtot)))
        if len(nan[0]) > 0:
            bad_starx = xpos[nan]
            bad_stary = ypos[nan]

            print(f'image {ind} NaN fluxerr star (x,y): {bad_starx, bad_stary}')

        #append to list
        ePhotometry.append(errtot)

    ePhotometry = np.array(ePhotometry)
    return ePhotometry



# detrend all stars
def detrend(photometry, ephotometry, nstars):
    '''
    detrend

    detrend the background for each night so we don't have to worry about changes in background noise levels

    inputs
    -------
    photometry: list - list of flux values from aperture photometry
    ephotometry: list - list of flux errors from aperture photometry
    nstars: float - number of stars in the field

    outputs
    --------
    finalPhot: list - final aperture photometry of sources with bad sources replaced with nans.
                        << this is the list you want to use from now on. >>
    cPhotometry: list - detrended aperture photometry

    '''

    # replace all flux negative values with nans
    photometry = np.array(photometry)
    photometry[photometry<=0] = np.nan

    #create empty lists
    cPhotometry = np.zeros_like(photometry)
    finalPhot = np.zeros_like(photometry)

    checkstars = np.linspace(0,nstars,10).astype(int)



    # get median flux value for each star (find percent change)
    for star in np.arange(nstars):

        if star in checkstars:
            print('starting detrending on star number ', star, '/', nstars)
        ## pull out each star's photometry + error Photometry for each night and place in list
        starPhotometry = np.array([starPhot[star] for starPhot in photometry])
        starPhotometryerr = np.array([starPhoterr[star] for starPhoterr in ephotometry])

        #find signal to noise
        SNval = starPhotometry /starPhotometryerr

        #find all low S/N
        low_sn = np.where(SNval < 3.)

        if len(low_sn[0]) > 0:
            print(star, ': low S/N, night', low_sn[0])
        if star == 135:
            print(low_sn)

        # blank out bad photometry
        starPhotometry[low_sn] = np.nan

        # add final photometric values to list
        finalPhot[:,star] = starPhotometry

        # now find the median

        med_val = np.nanmedian(starPhotometry)
        if star in checkstars:
            print(f'median photometry on star {star}: {med_val}')

        if med_val <= 0.0: # known bad photometry
            cPhotometry[:,star] = starPhotometry*np.nan
        else:
            cPhotometry[:,star] = starPhotometry/med_val

    # do a check for outlier photometry?

    for night in np.arange(len(cPhotometry)):
        # remove large-scale image-to-image variation to find best stars
        cPhotometry[night] = cPhotometry[night]/np.nanmedian(cPhotometry[night])

    cPhotometry = np.array(cPhotometry)
    # eliminate stars with outliers from consideration
    w = np.where( (cPhotometry < 0.5) | (cPhotometry > 1.5))
    cPhotometry[w] = np.nan

    return finalPhot, cPhotometry


def plotPhotometry(Times,cPhotometry):
    '''plot detrended photometry'''
    plt.figure()
    for ind in np.arange(np.shape(cPhotometry)[1]):
        plt.scatter(Times-np.nanmin(Times),cPhotometry[:,ind],s=1.,color='black')


    # make the ranges a bit more general

    #plt.ylim(-1.,4.)
    plt.xlim(-0.1,1.1*np.max(Times-np.nanmin(Times)))

    plt.ylim(np.nanmin(cPhotometry),np.nanmax(cPhotometry))

    plt.xlabel('Observation Time [days]')
    plt.ylabel('Detrended Flux')
    plt.show()


def CaliforniaCoast(Photometry,cPhotometry,comp_num=9,flux_bins=6):
    """
    Find the least-variable stars as a function of star brightness*

    (it's called California Coast because the plot looks like California and we are looking for edge values: the coast)


    inputs
    --------------
    Photometry  : input Photometry catalog
    cPhotometry : input detrended Photometry catalog
    flux_bins   : (default=10) maximum number of flux bins
    comp_num    : (default=5)  minimum number of comparison stars to use


    outputs
    --------------
    BinStars      : dictionary of stars in each of the flux partitions
    LeastVariable : dictionary of least variable stars in each of the flux partitions


    """

    tmpX = np.nanmedian(Photometry,axis=0)
    tmpY = np.nanstd(cPhotometry, axis=0)

    xvals = tmpX[(np.isfinite(tmpX) & np.isfinite(tmpY))]
    yvals = tmpY[(np.isfinite(tmpX) & np.isfinite(tmpY))]
    kept_vals = np.where((np.isfinite(tmpX) & np.isfinite(tmpY)))[0]
    #print('Keep',kept_vals)

    # make the bins in flux, equal in percentile
    flux_percents = np.linspace(0.,100.,flux_bins)
    print('Bin Percentiles to check:',flux_percents)

    # make the dictionary to return the best stars
    LeastVariable = {}
    BinStars = {}

    for bin_num in range(0,flux_percents.size-1):

        # get the flux boundaries for this bin
        min_flux = np.percentile(xvals,flux_percents[bin_num])
        max_flux = np.percentile(xvals,flux_percents[bin_num+1])
        #print('Min/Max',min_flux,max_flux)

        # select the stars meeting the criteria
        w = np.where( (xvals >= min_flux) & (xvals < max_flux))[0]

        BinStars[bin_num] = kept_vals[w]

        # now look at the least variable X stars
        nstars = w.size
        #print('Number of stars in bin {}:'.format(bin_num),nstars)

        # organize stars by flux uncertainty
        binStarsX = xvals[w]
        binStarsY = yvals[w]

        # mininum Y stars in the bin:
        lowestY = kept_vals[w[binStarsY.argsort()][0:comp_num]]

        #print('Best {} stars in bin {}:'.format(comp_num,bin_num),lowestY)
        LeastVariable[bin_num] = lowestY

    return BinStars,LeastVariable



def findComparisonStars(Photometry, cPhotometry, accuracy_threshold = 0.2, plot=True,comp_num=6): #0.025
    '''
    findComparisonStars*

    finds stars that are similar over the various nights to use as comparison stars

    inputs
    --------
    Photometry: list - photometric values taken from detrend() function.
    cPhotometry: list - detrended photometric values from detrend() function
    accuracy_threshold: float - level of accuracy in fluxes between various nights
    plot: boolean - True/False plot various stars and highlight comparison stars

    outputs
    --------
    most_accurate: list - list of indices of the locations in Photometry which have the best stars to use as comparisons
    '''

    BinStars,LeastVariable = CaliforniaCoast(Photometry,cPhotometry,comp_num=comp_num)

    star_err = np.nanstd(cPhotometry, axis=0)

    if plot:
        xvals = np.log10(np.nanmedian(Photometry,axis=0))
        yvals = np.log10(np.nanstd(cPhotometry, axis=0))

        plt.figure()
        plt.scatter(xvals,yvals,color='black',s=1.)
        plt.xlabel('log Median Flux per star')
        plt.ylabel('log De-trended Standard Deviation')
        plt.text(np.nanmin(np.log10(np.nanmedian(Photometry,axis=0))),np.nanmin(np.log10(star_err[star_err>0.])),\
            'Less Variable',color='red',ha='left',va='bottom')
        plt.text(np.nanmax(np.log10(np.nanmedian(Photometry,axis=0))),np.nanmax(np.log10(star_err[star_err>0.])),\
            'More Variable',color='red',ha='right',va='top')

        for k in LeastVariable.keys():
            plt.scatter(xvals[LeastVariable[k]],yvals[LeastVariable[k]],color='red')


    # this is the middle key for safety
    middle_key = np.array(list(LeastVariable.keys()))[len(LeastVariable.keys())//2]

    # but now let's select the brightest one
    best_key = np.array(list(LeastVariable.keys()))[-1]

    return LeastVariable[best_key]





#def findComparisonStars_sarah(Photometry, cPhotometry, accuracy_threshold = 0.2, plot=True): #0.025
#    '''
#    findComparisonStars
#
#    finds stars that are similar over the various nights to use as comparison stars
#
#    inputs
#    --------
#    Photometry: list - photometric values taken from detrend() function.
#    cPhotometry: list - detrended photometric values from detrend() function
#    accuracy_threshold: float - level of accuracy in fluxes between various nights
#    plot: boolean - True/False plot various stars and highlight comparison stars
#
#    outputs
#    --------
#    most_accurate: list - list of indices of the locations in Photometry which have the best stars to use as comparisons
#    '''
#
#    #find std of detrended photometry
#    star_err = np.nanstd(cPhotometry, axis=0)
#
#    #determine stars that do not vary of each image.
#    most_accurate = np.where((star_err < accuracy_threshold) & (star_err > 0.) &                              (np.nanmedian(Photometry, axis=0) < 3000.) & (np.nanmedian(Photometry, axis=0) >100.) )[0]
#
#    print('number of good comparison stars: ', len(most_accurate), ': index = ', most_accurate)
#    print('avg flux values of comparison stars: ', np.log10(np.nanmedian(Photometry,axis=0))[most_accurate],'+/-', np.log10(star_err)[most_accurate])
#
#    #plot good comparison stars in red, everything else in black
#    if plot:
#        plt.figure()
#        plt.scatter(np.log10(np.nanmedian(Photometry,axis=0)),np.log10(star_err),color='black',s=1.)
#        plt.scatter(np.log10(np.nanmedian(Photometry,axis=0))[most_accurate],np.log10(star_err)[most_accurate],color='red',s=10.)
#        plt.xlabel('log Flux',size=24)
#        plt.ylabel('log Uncertainty',size=24)
#
#        plt.xlim(-1.,6)
#
#    return most_accurate


def runDifferentialPhotometry(Photometry, ePhotometry, nstars, most_accurate):

    '''
    runDifferentialPhotometry

    as the name says!

    inputs
    ----------
    Photometry: list - list of photometric values from detrend() function
    ePhotometry: list - list of photometric error values
    nstars: float - number of stars
    most_accurate: list - list of indices of non variable comparison stars

    outputs
    ---------
    dPhotometry: list - differential photometry list
    edPhotometry: list - scaling factors to photometry error
    tePhotometry: list - differential photometry error

    '''

    #number of nights of photometry
    nimages = len(Photometry)

    #range of number of nights
    imgindex = np.arange(0,nimages,1)

    #create lists for diff photometry
    dPhotometry = np.zeros([nimages,len(Photometry[0])])
    edPhotometry = np.zeros([nimages,len(Photometry[0])])
    eedPhotometry = np.zeros([nimages,len(Photometry[0])])
    tePhotometry = np.zeros([nimages,len(Photometry[0])])

    checkstars = np.linspace(0,nstars,10).astype(int)



    for star in range(0,nstars):

        if star in checkstars:

            print('running differential photometry on star: ', star, '/', nstars)

        #pull out each star's photometry + error Photometry for each night and place in list
        starPhotometry = np.array([starPhot[star] for starPhot in Photometry])
        starPhotometryerr = np.array([starPhoterr[star] for starPhoterr in ePhotometry])

        #create temporary photometry list for each comparison star
        tmp_phot = np.zeros([nimages,len(most_accurate)])

        #go through comparison stars and determine differential photometry
        for ind,comparison_index in enumerate(most_accurate):
            #pull out comparison star's photometry for each night and place in list
            compStarPhotometry = np.array([starPhot[comparison_index] for starPhot in Photometry])

            #calculate differential photometry
            tmp_phot[:,ind] = (starPhotometry*np.nanmedian(compStarPhotometry))/(compStarPhotometry*np.nanmedian(starPhotometry))

        #median combine differential photometry found with each comparison star for every other star
        dPhotometry[:,star] = np.nanmedian(tmp_phot,axis=1)

        # apply final scaling factors to the photometric error
        edPhotometry[:,star] = starPhotometryerr*(np.nanmedian(tmp_phot,axis=1)/starPhotometry)

        # the differential photometry error
        eedPhotometry[:,star] = np.nanstd(tmp_phot,axis=1)

        # the differential photometry error
        tePhotometry[:,star] = ((starPhotometryerr*(np.nanmedian(tmp_phot,axis=1)/starPhotometry))**2. + (np.nanstd(tmp_phot,axis=1))**2.)**0.5

    return dPhotometry, edPhotometry, tePhotometry


def target_list(memberlist, ra_all, dec_all, max_sep=20.0):
    #checks to see if memberlist is a tuple or region file
    if isinstance(memberlist, tuple):
        ra_mem = [memberlist[0]]
        dec_mem = [memberlist[1]]
    elif isinstance(memberlist, str):
        try:
            regions = read_ds9(memberlist)
            ra_mem = [i.center.ra.deg for i in regions]
            dec_mem = [i.center.dec.deg for i in regions]
        except:
            print('memberlist must be a region file or tuple')

    else:
        ValueError('memberlist must be region file or tuple')


    #finds your target star index in the catalog found with DAOStarFinder
    c = SkyCoord(ra=ra_mem*u.degree, dec=dec_mem*u.degree)
    catalog = SkyCoord(ra=ra_all*u.degree, dec=dec_all*u.degree)
#    idx, d2d, d3d = c.match_to_catalog_sky(catalog) #idx is the index in ra_all, dec_all where your target star is located

    max_sep = max_sep * u.arcsec
    ind, d2d, d3d = c.match_to_catalog_3d(catalog)
    sep_constraint = d2d < max_sep
#    c_matches = c[sep_constraint]
#    catalog_matches = c[ind[sep_constraint]]

    idx = ind[sep_constraint]

#    print(ra_mem)
#
#    print(ra_all[idx])
#    print( dec_mem)
#    print(dec_all[idx])

    return idx, ra_all[idx], dec_all[idx]


def diffPhot_IndividualStars(datadir, memberlist, ra_all, dec_all, xpos, ypos, dPhotometry, edPhotometry, tePhotometry, times, target, fitsimage, most_accurate,verbose=1, max_sep=20.0):
    '''
    diffPhot_IndividualStars

    pull out differential photometry for objects of interest from region file

    inputs
    --------
    memberlist: tuple OR region file
         can either be a tuple (<RA>, <DEC>)  or a region file listing ra and dec of sources
    ra_all, dec_all, xpos, ypos: list - list of ra, dec, x pixel, y pixel positions of all stars in field
    dPhotometry, edPhotometry, eedPhotometry, tePhotometry: lists - lists of differential, photometric err, differential photometric error fluxes found with runDifferentialPhotometry() function
    times: list - list of time stamps
    target: string - name of target

    outputs
    --------
    npz save file: numpy save file with ra, dec, xpos, ypos, time, phase, period, flux, fluxerr, and power for each target star

    can be read back in by:
    data = np.load('<name of file>.npz')
    # get column names
    print(data.files)
    # read ra positions of all stars
    print(data['ra'])

    '''

    #checks to see if memberlist is a tuple or region file
    if isinstance(memberlist, tuple):
        ra_mem = [memberlist[0]]
        dec_mem = [memberlist[1]]
    elif isinstance(memberlist, str):
        try:
            regions = read_ds9(memberlist)
            ra_mem = [i.center.ra.deg for i in regions]
            dec_mem = [i.center.dec.deg for i in regions]
        except:
            print('memberlist must be a region file or tuple')

    else:
        ValueError('memberlist must be region file or tuple')


    #finds your target star index in the catalog found with DAOStarFinder
    c = SkyCoord(ra=ra_mem*u.degree, dec=dec_mem*u.degree)
    catalog = SkyCoord(ra=ra_all*u.degree, dec=dec_all*u.degree)
#    idx, d2d, d3d = c.match_to_catalog_sky(catalog) #idx is the index in ra_all, dec_all where your target star is located

    max_sep = max_sep * u.arcsec
    ind, d2d, d3d = c.match_to_catalog_3d(catalog)
    sep_constraint = d2d < max_sep
#    c_matches = c[sep_constraint]
#    catalog_matches = c[ind[sep_constraint]]

    idx = ind[sep_constraint]

#    print(ra_mem)
#
#    print(ra_all[idx])
#    print( dec_mem)
#    print(dec_all[idx])
#

    print('number of target stars:', len(idx))
    print('index of target stars:', idx)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(fits.getdata(fitsimage))
    plt.figure(figsize=(10,10))
    plt.imshow(fits.getdata(fitsimage), vmin=vmin, vmax=vmax)
    plt.plot(xpos[idx], ypos[idx], 'ro', markersize=10)
    plt.scatter(xpos[most_accurate], ypos[most_accurate], s=70, linewidth=3, facecolors='none', edgecolors='magenta')
    plt.plot(xpos[most_accurate], ypos[most_accurate], 'kx')
    plt.title('Target Stars and Comparisons')
    plt.show()



    #determine time
    time = np.array(times - np.nanmin(times))
    #pulls out differential photometry of target stars
    flux = np.array(dPhotometry[:,idx])
    #pulls out differetial photometry error of target stars
    fluxerr = np.array(tePhotometry[:,idx])

    #create empty arrays for time, flux, fluxerr, phase, period, power for target stars
    # time = []
    # flux = []
    # fluxerr = []
    # foldedphase = []
    # totperiod  = []
    # totpower = []

    #run periodogram, phase folding for each target star.
    # for star in np.arange(len(idx)):
    #     print('starting analysis on star at: (x,y) = (', xpos[star], ypos[star]), ')'

    #     #get photometry and time for each target star
    #     TT = totalTime
    #     DD = totalDiff[:,star]
    #     ED = totalerrDiff[:,star]

    #     #determine where the values are good.
    #     w = np.where(np.isfinite(TT) & np.isfinite(DD) & np.isfinite(ED))

    #     if verbose>1:
    #         print(TT,DD, ED)
    #         print(TT[w], DD[w], ED[w])
    #     #run LombScargle
    #     ls = LombScargle(TT[w], DD[w], ED[w])

    #     #get power and frequency
    #     try:
    #         frequency, power = ls.autopower()
    #     except:
    #         print('failed.  If memberlist was a tuple -- try more sig figs')


    #     #rename to make things easier to read :)
    #     diffFluxerr = ED
    #     diffFlux = DD

    #     #determine period
    #     per = 1./frequency

    #     #determine best frequency from LombScargle
    #     w = np.where( (per > 0.2) & (per< 4.))
    #     best_frequency = frequency[w][np.argmax(power[w])]

    #     #phase folding
    #     newtime = TT % (1./best_frequency)
    #     phase = newtime - np.round(newtime,0) + 0.5

    #     #append all to list
    #     time.append(TT)
    #     flux.append(diffFlux)
    #     fluxerr.append(diffFluxerr)
    #     foldedphase.append(phase)
    #     totperiod.append(per)
    #     totpower.append(power)

        #can save as .csv file-- however, all lists save as strings so it is hard to use
#     d = {'ra':catalog[idx].ra.deg, 'dec':catalog[idx].dec.deg, 'xpos':xpos[idx],'ypos':ypos[idx], 'time': time,\
#          'foldedphase': foldedphase, 'period':totperiod, 'differentialFlux': flux,\
#          'differentialFluxerr': fluxerr, 'power': totpower}

#     df = pd.DataFrame(d)
#     df.name = target

#     print(df)
#     df.to_csv('differentialPhot_field' + target + '.csv' , sep=';')


    savefile = datadir + 'differentialPhot_field' + target
    print('finished. Saving catalog to:', savefile)

    #save as npz file.  much easier and saves lists as lists
    #np.savez(savefile, ra = catalog[idx].ra.deg, dec=catalog[idx].dec.deg, xpos=xpos[idx], ypos=ypos[idx], time=time,foldedphase =foldedphase, period=totperiod, flux=flux,fluxerr= fluxerr, power=totpower  )
    a = np.savez(savefile, ra = catalog[idx].ra.deg, dec=catalog[idx].dec.deg, xpos=xpos[idx], ypos=ypos[idx], time=time, flux=flux,fluxerr= fluxerr)


    # return catalog[idx].ra.deg, catalog[idx].dec.deg, xpos[idx], ypos[idx], time, foldedphase, totperiod, flux, fluxerr, totpower
    return catalog[idx].ra.deg, catalog[idx].dec.deg, xpos[idx], ypos[idx], time, flux, fluxerr
