# MDGSC reduction pipeline
# by William Balmer

# Data reduction pipeline

# most functions and general architecture based (or lifted outright) on work
# passed down from Sarah Betti (2020) and Kim Ward-Duong (2018-2019) for courses
# ASTR 337 and 341 in the Five College Astronomy Department.
# Many thanks to them for being such awesome astronomy instructors!

# Updated subpixel shift function based on this scikit-image example:
# https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html?highlight=shift
# and course content from Laurent Pueyo and Anand Sivaramakrishnan's Fourier
# Optics and Interferometry course at JHU.

### Imports ###

# general
import os
import glob
import time
import tqdm

# numerical
import numpy as np

# scipy
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import shift

# scikit-image
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft

# plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# astropy
from astropy.io import fits
from astropy import stats
from astropy.stats import sigma_clip
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans, convolve

# for RGB functions
from PIL import Image as PILIMG


### Reduction functions (general) ###

def mediancombine(filelist):
    '''
    median combine image stack function
    '''
    n = len(filelist)
    first_frame_data = fits.getdata(filelist[0])
    imsize_y, imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y, imsize_x , n))
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im
    med_frame = np.median(fits_stack, axis = 2)
    return med_frame


def bias_subtract(filename, path_to_bias, outpath, save=True):
    '''
    bias subtraction function
    '''
    targetdata = fits.getdata(filename)
    target_header = fits.getheader(filename)
    biasdata = fits.getdata(path_to_bias)

    b_data = targetdata - biasdata

    fitsname = filename.split('/')[-1]
    if save:
        fits.writeto(outpath + "/" + 'b_' + fitsname, b_data, target_header, overwrite=True)
    else:
        return b_data


def dark_subtract(filename, path_to_dark, outpath, scale=False, save=True):
    '''
    performs dark subtraction on your flat/science fields.
    '''

    # open the flat/science field data and header
    frame_data = fits.getdata(filename)
    frame_header = fits.getheader(filename)

    #open the master dark frame with the same exposure time as your data.
    master_dark_data = fits.getdata(path_to_dark)

    #subtract off the dark current
    if scale:
        if frame_header['EXPTIME'] != fits.getheader(path_to_dark)['EXPTIME']:
            scale = frame_header['EXPTIME'] / fits.getheader(path_to_dark)['EXPTIME']
            master_dark_data = scale * master_dark_data

    dark_subtracted = frame_data - master_dark_data

    new_filename = filename.split('/')[-1]

    if save:
        fits.writeto(outpath + '/d' + new_filename, dark_subtracted, frame_header,overwrite=True)
    else:
        return dark_subtracted


def norm_combine_flats(filelist):
    '''
    normalize and combine frames
    '''
    n = len(filelist)
    first_frame_data = fits.getdata(filelist[0])
    imsize_y, imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y, imsize_x , n))
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        norm_im = im/np.median(im)
        fits_stack[:,:,ii] = norm_im

    med_frame = np.median(fits_stack, axis=2)
    return med_frame


def cross_image(im1, im2, xcen=None, ycen=None, boxsize=None, upsample=3):
    '''
    cross_image
    ---------------
    implement DFT upsampled cross-correlation of two images to find offsets


    inputs
    ---------------
    im1                      : (matrix of floats)  first input image
    im2                      : (matrix of floats) second input image
    xcen                     : (integer, optional) x center of subregion of image to cross-correlate
    ycen                     : (integer, optional) y center of subregion of image to cross-correlate
    boxsize                  : (integer, optional) subregion of image to cross-correlate
    upsample                 : (float, optional) upsampling rate for subpixel shifts

    returns
    ---------------
    xshift                   : (float) x-shift in pixels
    yshift                   : (float) y-shift in pixels

    dependencies
    ---------------
    phase_cross_correlation   : from skimage.registration import phase_cross_correlation
    numpy                     : import numpy as np

    '''

    # The type cast into 'float' to avoid overflows:
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    if boxsize is not None:
        if xcen is not None:
            if ycen is not None:
                im1_gray = im1_gray[ycen-boxsize:ycen+boxsize,xcen-boxsize:xcen+boxsize]
                im2_gray = im2_gray[ycen-boxsize:ycen+boxsize,xcen-boxsize:xcen+boxsize]

    # guard against extra nan values
    im1_gray[np.isnan(im1_gray)] = np.nanmedian(im1_gray)
    im2_gray[np.isnan(im2_gray)] = np.nanmedian(im2_gray)

    # calculate shifts, errors, and phase offset
    shift, error, diffphase = phase_cross_correlation(im1_gray, im2_gray,
                                                      upsample_factor=100)
    # assign x and y shift variables to return
    yshift, xshift = shift
    # TODO: options to return errors? for the purposes of simple reg i haven't needed
    return xshift,yshift


def shift_image(image,xshift,yshift):
    '''
    shift_image
    -------------
    wrapper for scipy's implementation that shifts images according to values from cross_image

    inputs
    ------------
    image           : (matrix of floats) image to be shifted
    xshift          : (float) x-shift in pixels
    yshift          : (float) y-shift in pixels

    outputs
    ------------
    shifted image   : shifted, interpolated image.
                      same shape as input image, with zeros filled where the image is rolled over

    dependencies
    ---------------
    shift           : from scipy.ndimage.interpolation import shift
    '''
    return shift(image,(xshift,yshift))


def scale_filter(tmpimg,lowsig,highsig):
    '''
    scale images for combining into RGB image
    '''

    tmpimg -= np.median(tmpimg)
    tmpimg = gaussian_filter(tmpimg, sigma=(0.5, 0.5), order=0)
    print('minmax 1: ', np.min(tmpimg),np.max(tmpimg))


    tmpsig = stats.sigma_clipped_stats(tmpimg, sigma=2, maxiters=5)[2]
    print('std: ', tmpsig)
    print("lowsig, highsig: ", lowsig, highsig)
    print('cuts: ', lowsig*tmpsig, highsig*tmpsig)

    # apply thresholding
    tmpimg[np.where(tmpimg < lowsig*tmpsig)] = lowsig*tmpsig
    tmpimg[np.where(tmpimg > highsig*tmpsig)] = highsig*tmpsig
    print('minmax 2: ', np.min(tmpimg),np.max(tmpimg))

    # arcsin scaling
    tmpimg = np.arcsinh(tmpimg)
    print('minmax 3: ', np.min(tmpimg),np.max(tmpimg))

    # scale to [0,255]
    tmpimg += np.min(tmpimg)
    tmpimg *= 255./np.max(tmpimg)
    tmpimg[np.where(tmpimg < 0.)] = 0.
    print('minmax 4: ', np.min(tmpimg),np.max(tmpimg))

    # recast as unsigned integers for jpeg writer
    IMG = PILIMG.fromarray(np.uint8(tmpimg))

    print("")

    return IMG

### Pipeline functions ###

def run_master_bias(fil):
    '''
    create master bias
    '''
    if not os.path.exists(f'{fil}/Masters'):
        os.makedirs(f'{fil}/Masters')
        print(f'Masters folder created at: {fil}/Masters')

    masterbiaspath = fil + '/Masters/MasterBias.fit'
    if not os.path.exists(f'{fil}/Masters/MasterBias.fit'):
        print('Making Master Bias')
        # create master bias
        bias_fits = glob.glob(fil + '/bias/*.fit*')

        median_bias = mediancombine(bias_fits)
        fits.writeto(masterbiaspath, median_bias, header=fits.getheader(bias_fits[0]), overwrite=True)
    else:
        print(f'Master Bias in: {masterbiaspath}')
    print()


def run_master_dark(fil):
    '''
    create master darks for each exposure time
    '''
    masterbiaspath = fil + '/Masters/MasterBias.fit'
    masterdarkpath = fil + '/Masters/'

    darkmaster_test = glob.glob(f'{fil}/Masters/MasterDark*.fit*')
    if len(darkmaster_test) == 0:
        print('Making Master Darks')
        # create master dark
        dark_outpath = fil + '/darks'
        b_dark_test = glob.glob(fil + '/darks/b_*.fit*')
        for im in b_dark_test:
            os.remove(im)
        dark_fits = glob.glob(fil + '/darks/*.fit*')

        #bias subtract darks
        for darks in dark_fits:
            bias_subtract(darks, masterbiaspath, dark_outpath)

        # median combine bias subtracted dark frames with same exposure time
        b_dark_fits = glob.glob(fil + '/darks/b_*.fit*')

        # sort darks into folders based on exposure time
        for b_darks in b_dark_fits:
            exptime = fits.getheader(b_darks)['EXPTIME']
            filname = b_darks.split('/')[-1]

            if not os.path.exists(fil + '/darks/darks' + str(exptime)):
                os.makedirs(fil + '/darks/darks' +  str(exptime))

            if not os.path.exists(fil + '/darks/darks' + str(exptime) + '/' + filname):
                os.rename(b_darks,fil + '/darks/darks' +  str(exptime) + '/' + filname)

        # glob all folders
        b_dark_exptime_folder = glob.glob(fil + '/darks/darks*')

        for exp_folder in b_dark_exptime_folder:
            dark_time = exp_folder.split('/')[-1]
            time = dark_time.split('s')[-1]
            print(f'exposure time {time}')
            b_dark_exptime_fits = glob.glob(exp_folder + '/*.fit*')
            median_dark_exptime = mediancombine(b_dark_exptime_fits)
            print('path to dark: ' + masterdarkpath + 'MasterDark' + time + '.fit')
            fits.writeto(masterdarkpath + 'MasterDark' + time + '.fit', median_dark_exptime, header=fits.getheader(b_dark_exptime_fits[0]), overwrite=True)
    else:
        print(f'Master Darks: {darkmaster_test}')
    print()


def run_master_flat(fil):
    '''
    create master flats for each filter
    '''
    masterbiaspath = fil + '/Masters/MasterBias.fit'
    masterdarkpath = fil + '/Masters/'
    flatfield_test = glob.glob(f'{fil}/Masters/MasterFlat*.fits')
    if len(flatfield_test) == 0:
        print('Starting Flat fields')

        # bias subtract flat fields
        b_flat_test = glob.glob(fil + '/flats/b_*.fit*')
        for im in b_flat_test:
            os.remove(im)

        flat_files = glob.glob(fil + '/flats/*.fit*')
        for flats in flat_files:
            bias_subtract(flats, masterbiaspath, fil+'/flats')

        #dark subtract flat fields
        db_flat_test = glob.glob(fil+'/flats/*/db_*.fit*')
        for im in db_flat_test:
            os.remove(im)

        b_flat_files = glob.glob(fil + '/flats/b_*.fit*')

        for b_flats in b_flat_files:
            exptime = fits.getheader(b_flats)['EXPTIME']
            filters = fits.getheader(b_flats)['FILTER'][0]
            if os.path.exists(masterdarkpath + 'MasterDark' + str(exptime) + '.fit'):
                masterdark = masterdarkpath + 'MasterDark' + str(exptime) + '.fit'
            else:
                masterdark = glob.glob( masterdarkpath + 'MasterDark*.fit*')[-1]

            if not os.path.exists(fil + '/flats/' + filters + 'flat'):
                os.makedirs(fil + '/flats/' + filters + 'flat')

            dark_subtract(b_flats, masterdark, fil + '/flats/' + filters + 'flat')


        # norm combine flat fields
        flat_bands = glob.glob(fil + '/flats/*flat')
        for band in flat_bands:
            db_flats = glob.glob(band + '/db_*.fit*')

            norm_flat = norm_combine_flats(db_flats)

            flat_header = fits.getheader(db_flats[0])
            band_name = flat_header['FILTER'][0]

            print('path to '+ band_name + ' flat: ' + fil + '/Masters/MasterFlat_' + band_name  + '.fit')
            fits.writeto(fil + '/Masters/MasterFlat_' + band_name  + '.fit', norm_flat, flat_header, overwrite=True)
    else:
        print(f'Path to Master Flats: {flatfield_test}')
    print()


def run_targets(fil, targs, scaleDarks=True, saveSpace=True):
    '''
    bias, dark, and flat field science targets
    '''
    for target in tqdm.tqdm(targs):
        print()
        print('------------o------------')
        print('target: ', target)
        print()
        masterbiaspath = fil + '/Masters/MasterBias.fit'
        masterpath = fil + '/Masters/'

        # bias subtract targets
        bias_images = glob.glob(f'{fil}/{target}/b_*.fit*')
        scidata = glob.glob(fil + '/*' + target + '/*.fit*')

        filters = []

        if len(bias_images)!=len(scidata):
            [os.remove(im) for im in bias_images]
            print('Bias subtracting ')
            for sci_image in tqdm.tqdm(scidata):
                filtername = fits.getheader(sci_image)['FILTER'][0]
                sci_outpath = fil + '/' + target + '/' + filtername + 'band'
                if not os.path.exists(sci_outpath):
                    os.makedirs(sci_outpath)
                bias_subtract(sci_image, masterbiaspath, sci_outpath)
                filters.append(filtername)
        filters = np.unique(filters)

        # dark subtract bias targets
        for filtername in filters:
            b_scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/b_*.fit*')

            dark_images = glob.glob(f'{fil}/{target}/{filtername}band/db*.fit*')

            if len(dark_images)!=len(b_scidata):
                print('Dark subtracting ', filtername, ' band')
                [os.remove(im) for im in dark_images]
                sci_outpath = fil + '/' + target + '/' + filtername + 'band'
                for b_sci_image in tqdm.tqdm(b_scidata):
                    exptime = fits.getheader(b_sci_image)['EXPTIME']
                    if os.path.exists(masterpath + 'MasterDark' + str(exptime) + '.fit'):
                        masterdark = masterpath + 'MasterDark' + str(exptime) + '.fit'
                    else:
                        masterdark = glob.glob( masterpath + 'MasterDark*.fit*')[-1]

                    dark_subtract(b_sci_image, masterdark, sci_outpath, scale=scaleDarks)


        # flat field db targets
        for filtername in filters:
            db_scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/db_*.fit*')
            flat_images = glob.glob(f'{fil}/{target}/{filtername}band/fdb*.fit*')

            if len(flat_images)!=len(db_scidata):

                [os.remove(im) for im in flat_images]
                masterflat = masterpath + '/MasterFlat_' + filtername + '.fit'
                masterflat_data = fits.getdata(masterflat)

                sci_outpath = fil + '/' + target + '/' + filtername + 'band'

                for db_sci_image in tqdm.tqdm(db_scidata):
                    db_sci_data = fits.getdata(db_sci_image)
                    db_sci_hdr = fits.getheader(db_sci_image)

                    fdb_sci_image = db_sci_data / masterflat_data
                    sci_name = db_sci_image.split('/')[-1]
                    fits.writeto(sci_outpath + '/f' + sci_name, fdb_sci_image, db_sci_hdr, overwrite=True )

        if saveSpace:
            for filtername in filters:
                print('Deleting b_ and db_ files for ', filtername, ' band')
                b_scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/b_*.fit*')
                db_scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/db_*.fit*')
                for file in b_scidata:
                    if os.path.isfile(file):
                        os.remove(file)
                for file in db_scidata:
                    if os.path.isfile(file):
                        os.remove(file)
    print()
    print('------------o------------')
    print('Done reducing all targets!')
    return


def flip_images(filenames):
    '''
    rotate images if meridian was flipped (not sure if we need this, vestigial to Amherst)
    '''
    for i in filenames:
        im = fits.getdata(i)
        #im_flip = np.flip(im, axis=(1,0))
        im_flip = np.flip(im, axis=(0,1))
        head = fits.getheader(i)
        fits.writeto(i.replace('.fit', '_f.fit'), im_flip, header=head, overwrite=True)


def run_register_align(datadir, targs, filters, centerx=None, centery=None, boxsize=1400, refim=None,
                       saveIndivs=False, saveStack=True):
    '''
    register images
    '''
    if isinstance(targs, str):
        targs = [targs]
    if isinstance(centerx, int):
        centerx = [centerx]
        centery = [centery]

    if centerx==None:
        centerx = np.ones_like(targs).astype(int) * 2048
    if centery==None:
        centery = np.ones_like(targs).astype(int) * 2048

    # Cycle through list of targets:
    for ind, targname in enumerate(targs):
        print(' ')
        print('-----------------------------')
        print('target: ', targname)
        print('-----------------------------')

        # Using glob, make list of all reduced images of current target in all filters.
        print(datadir + '/' + targname + '/*band/fdb*.fit*')
        imlist = glob.glob(datadir + '/' + targname + '/*band/fdb*.fit*')

        # Check to make sure that your new list has the right files:
        print("All files to be aligned: \n", imlist)
        print()
        print(len(imlist), ' files to be aligned')
        print('\n')

        # Open image that others will be shifted to (default is first image)
        if refim is not None:
            im1,hdr1 = fits.getdata(imlist[refim],header=True)
            kernel = Gaussian2DKernel(x_stddev=1)
            im1 = interpolate_replace_nans(im1, kernel)
            print("Aligning all images to:", imlist[0])
        else:
            im1,hdr1 = fits.getdata(imlist[0],header=True)
            kernel = Gaussian2DKernel(x_stddev=1)
            im1 = interpolate_replace_nans(im1, kernel)
            print("Aligning all images to:", imlist[0])

        print('\n')

        xshifts = {}
        yshifts = {}

        for index,filename in enumerate(imlist):
            im,hdr = fits.getdata(filename,header=True)
            kernel = Gaussian2DKernel(x_stddev=0.1, y_stddev=0.1)
            im = interpolate_replace_nans(im, kernel)
            xshifts[index], yshifts[index] = cross_image(im1, im, centerx[ind], centery[ind], boxsize=boxsize)
            print("Shift for image", index, "is", xshifts[index], yshifts[index])

        # Calculate trim edges of new median stacked images so all stacked images of each target have same size
        max_x_shift = int(np.max([abs(xshifts[x]) for x in xshifts.keys()]))
        max_y_shift = int(np.max([abs(yshifts[x]) for x in yshifts.keys()]))

        print('   Max x-shift={0}, max y-shift={1} (pixels)'.format(max_x_shift,max_y_shift))

        # Cycle through list of filters
        for filtername in filters:
            # Create a list of FITS files matching *only* the selected filter:
            print('looking for files in '+datadir + '/' + targname + '/' + filtername + 'band/fdb*.fit*')
            scilist = glob.glob(datadir + '/' + targname + '/' + filtername + 'band/fdb*.fit*')

            if len(scilist) < 1:
                print("Warning! No files in scilist. Your path is likely incorrect.")
                break

            nfiles = len(scilist)
            print('Stacking ', nfiles, filtername, ' science frames')

            # Define new array with same size as master image
            image_stack = np.zeros([im1.shape[0],im1.shape[1],len(scilist)])

            xshifts_filt = {}
            yshifts_filt = {}

            # Make a new directory in your datadir for the new shifted/stacked fits files
            if os.path.isdir(datadir + '/Shifted') == False:
                os.makedirs(datadir + '/Shifted')
                print('\n Making new subdirectory for shifted images:', datadir + '/Shifted \n')
            if os.path.isdir(datadir + '/Stacked') == False:
                os.makedirs(datadir + '/Stacked')
                print('\n Making new subdirectory for Stacked images:', datadir + '/Stacked \n')

            # print statements based on save configuration
            if saveIndivs:
                print('   Writing FITS files ',targname + '_' + filtername + '_registered_'+str(index)+'.fits', 'to ',datadir + '/Shifted/','\n')

            for index,filename in tqdm.tqdm(enumerate(scilist)):
                im,hdr = fits.getdata(filename,header=True)
                kernel = Gaussian2DKernel(x_stddev=0.1, y_stddev=0.1)
                im = interpolate_replace_nans(im, kernel)
                im = convolve(im, Gaussian2DKernel(x_stddev=0.5, y_stddev=0.5))
                xshifts_filt[index], yshifts_filt[index] = cross_image(im1, im, centerx[ind], centery[ind], boxsize=boxsize)
                print(xshifts_filt[index], yshifts_filt[index])
                image_stack[:,:,index] = shift_image(im,xshifts_filt[index], yshifts_filt[index])
                if saveIndivs:
                    fits.writeto(datadir + '/Shifted/' + targname + '_' + filtername + '_registered_'+str(index)+'.fits', image_stack[:,:,index], fits.getheader(scilist[index]), overwrite=True)

            if saveStack:
                median_image = np.nanmedian(image_stack,axis=2)
                # Save the final stacked images into your new folder:
                fits.writeto(datadir + '/Stacked/' + targname + '_' + filtername + 'stack.fits', median_image, fits.getheader(scilist[0]), overwrite=True)
                print('   Wrote FITS file ',targname+'_'+filtername+'stack.fits', 'in ',datadir + '/Stacked/','\n')

    print(' ')
    print('-----------------------------')
    print('\n Done registering images!')
    return

def run_RGB(datadir, targname, filters, siglowhi):
    '''
    create RGB image
    ex. siglowhi = [-2,10.,-5,15.,-2,11.]
    BGR low and high sigma limits.
    To make red brighter, make 6th # lower.
    '''

    # Read in 3 stacked images
    Rtmp = fits.getdata(datadir+'/Stacked/'+targname+'_'+filters[2]+'stack.fits')
    Gtmp = fits.getdata(datadir+'/Stacked/'+targname+'_'+filters[1]+'stack.fits')
    Btmp = fits.getdata(datadir+'/Stacked/'+targname+'_'+filters[0]+'stack.fits')

    # Scale all 3 images
    print('Calculating stats....')
    R = scale_filter(Rtmp,lowsig=siglowhi[4],highsig=siglowhi[5])
    G = scale_filter(Gtmp,lowsig=siglowhi[2],highsig=siglowhi[3])
    B = scale_filter(Btmp,lowsig=siglowhi[0],highsig=siglowhi[1])

    # Merge 3 images into one RGB image
    im = PILIMG.merge("RGB", (R,G,B))

    im.save(datadir+'/Stacked/'+targname+'_RGB.jpg', "JPEG")
    print("Saved image as ", datadir+'/Stacked/'+targname+'_RGB.jpg')
