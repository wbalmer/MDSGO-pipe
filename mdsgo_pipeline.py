# MDGSO reduction pipeline
# by William Balmer (2021)

# User input script

# general architecture based on work passed down from Sarah Betti (2020)
# and Kim Ward-Duong (2018-2019) for courses ASTR 337 and 341 in the Five
# College Astronomy Department. Many thanks to them for being such
# awesome astronomy instructors!

### Imports ###

from sort_filter import *
from reduction_pipeline import *

### Pipeline ###

# 1) Set path to raw data
fil = 'TESTDATA'

# 2) make a list of the names in the filenames of your data to sort
sort_target_names = [
                     # 'hd189733',
                     'ring_nebula',
                     'hd189733',
                     ]

# 3) make lists of the broadband and narrowband filters you observed in
filters = ['1']
nbfilters = ['5', '6', '7']

# 4) make a list of the targets you want to run the data reduction on
# targets in broadband (either L or RGB)
targs = [
         'hd189733',
         ]
# targets in narrowband (OIII, Ha, SII)
nbtargs = [
           'ring_nebula',
          ]
# 5) if you took logs, make a string that lists all the bad images you want to avoid
bad_frame_numbers = None # looks something like '63, 96, 107-120' or None


# 6) sort files
run_filesort(fil, sort_target_names, bad_frame_numbers=bad_frame_numbers)

# 7) choose which binning you'll reduce
fil = fil+'/bin3'

# 8) construct calibration master frames if you took calibration images
# otherwise, you'll get a FileNotFound error, so you'll have to put a previous
# "Masters" folder in the "fil" path and hope you haven't accumulated too much
# dust since the last calibrations :P
# run_master_bias(fil)
# run_master_dark(fil)
# run_master_flat(fil)

# 9) reduce the data
run_targets(fil,targs)
run_targets(fil,nbtargs)

# 10) create a list of tuples that contain the x, y and boxsize you'll register
# your images on. create a 3 tuple of all None if you want to register using the
# whole image (may be slow)
cens = [(800, 575, 500),]
cendict = {}
i = 0
for targ in targs:
    cendict[targ] = cens[i]
    i += 1

nbcens = [(80, 645, 25),]
nbcendict = {}
i = 0
for targ in nbtargs:
    nbcendict[targ] = cens[i]
    i += 1

# 11) assuming you want to register your broadband but save the individual files
# (for something like taking a transit observation) and register but stack the
# narrowband images (for something like an astrophoto), loop through the targs
# and run the "run_register_align" function with the appropriate saveIndivs or
# saveStack keywords. Use the "run_RGB" function to create a basic 3 color image

for targ in targs:
    # register images
    run_register_align(fil, targ, [filters[0]], centerx=cendict[targ][0],
                        centery=cendict[targ][1], boxsize=cendict[targ][2],
                        saveIndivs=True, saveStack=False)

for targ in nbtargs:
    # register images
    run_register_align(fil, targ, [nbfilters[2],nbfilters[1],nbfilters[0]],
                        centerx=nbcendict[targ][0], centery=nbcendict[targ][1],
                        boxsize=nbcendict[targ][2], saveIndivs=False,
                        saveStack=True)
    # create RGB image
    run_RGB(fil, targ, nbfilters, siglowhi = [-1,10,-1,10.,-1,10])
