import numpy as np
from os import path, listdir
import matplotlib.pyplot as plt
import logging as log
import ffmpeg
import traceback
import time

def ser_files(directory):
    """Finds all the .ser files within a directory.
    It requires os.path and os.listdir libraries.

    Parameters:
    directory (string): The path to look for .ser files.

    Returns:
    list of str: Array of strings with the name of each .ser file."""

    files = [f for f in listdir(directory) if path.isfile(path.join(directory, f))]
    new_files = []
    for i in range(len(files)):
        if files[i][-4:] == '.ser':
            new_files.append(files[i])
    return new_files

def parsing_video(vfile):
    """Reads a .ser video file containing various RoboDIMM frames.
    It requires matplotlib.pyplot (as plt), logging.log, ffmpeg and traceback libraries.

    Parameters:
    vfile (string): The file path.

    Returns:
    dict: Dictionary containing:
        - 'file': Path to file
        - 'name': File name
        - 'objalt': Altitude of object (in degrees)
        - 'object': Object observed
        - 'obstime': Time of observation
        - 'nframes': Number of frames contained
        - 'frames': Array of frames, being a 2D graylevel array each one"""

    log.info("Parsing '%s' video file", vfile)
    try:
        # Get video dimensions
        probe = ffmpeg.probe(vfile)
        stream = next((_stream for _stream in probe['streams']
                            if _stream['codec_type'] == 'video'), None)
        # Read video
        out, error = (ffmpeg
                .input(vfile, threads=16)
                .output("pipe:", format=stream['codec_name'], pix_fmt=stream['pix_fmt'],
                        loglevel="error")
                .run(capture_stdout=True))

        vid = {}
        vid['file'] = vfile
        vid['name'] = path.basename(vid['file'])
        vid['objalt'] = float(vid['name'].split('_')[0])
        vid['object'] = vid['name'].split('_')[1]
        vid['obstime'] = vid['name'].split('_')[-1].strip('.ser')
        vid['nframes'] = stream['nb_frames']
        # Convert buffer to NumPy array
        vid['frames'] = (np.frombuffer(out, np.uint16).
                        reshape([-1, stream['height'], stream['width']]))
        #~ plt.imshow(vid['frames'][5], cmap='Spectral_r')
        #~ plt.show()
    except:
        # print(traceback.format_exc())
        # print(f"ERROR parsing video, skipped!")
        return False

    return vid

def x_centroid(im):
    """Gives the center of mass position of a graylevel image.
    It requires numpy library as np.

    Parameters:
    im (2D array): Graylevel image

    Returns:
    list of float: x and y coordinates of the center of mass, in pixel units"""

    L_x = len(im[0,:])
    L_y = len(im[:,0])

    # Sum total intensity
    I_G = np.sum(im)

    # Integral
    xx = np.ones((L_y,L_x),dtype=np.uint16)*np.linspace(0,L_x-1,L_x,dtype=np.uint16)
    Integ_x = np.sum(xx*im)
    
    yy = np.ones((L_x,L_y),dtype=np.uint16)*np.linspace(0,L_y-1,L_y,dtype=np.uint16)
    yy = yy.transpose(1,0)
    Integ_y = np.sum(yy*im)
    
    return (Integ_x/I_G, Integ_y/I_G)

def x_win_centroid(im, r, winMat=False):
    """Gives the center of mass position within a window around the brightest pixel.
    It requires numpy library as np and x_centroid function to be defined.

    Parameters:
    im (2D array): Graylevel image
    r (float): Radius of the window
    winMat (bool): Whether to return the window matrix or just the coordinates values

    Returns:
    list of float: Brightest centroid coordinates
    if winMat:
        2D array: Binary image locating the window used to find the centroid"""

    y0, x0 = np.where(im==np.max(im))
    y0, x0 = y0[0], x0[0]

    window_im = np.zeros(np.shape(im),dtype=np.uint32)
    for j in range(np.shape(im)[0]):
        for i in range(np.shape(im)[1]):
            l = np.sqrt((j-y0)**2+(i-x0)**2)
            if int(l) <= r:
                window_im[j,i] = im[j,i]
    
    if winMat:
        return x_centroid(window_im), window_im > 0
    else:
        return x_centroid(window_im)

def del_cosmics(im1, th=0.5):
    """Deletes any isolated pixel after applying a threshold
    It requires numpy library as np.

    Parameters:
    im1 (2D array): Graylevel image
    th (float): Threshold to binarize the image

    Returns:
    2D array: Graylevel image without cosmic ray pixels"""

    im = (im1/np.max(im1))*65535
    th = th*65535
    im = im>th

    im_cosmics = np.ones((len(im[:,0]), len(im[0,:])))
    for j in range(1,len(im[0,:])-1):
        for i in range(1,len(im[:,0])-1):
            if im[i,j] == 1 and im[i+1,j] == 0 and im[i-1,j] == 0 and im[i,j+1] == 0 and im[i,j-1] == 0:
                im_cosmics[i,j] = 0

    return im1*im_cosmics
    
def frame_cent(im, r):
    """Finds both centroids in the image.
    It requires del_cosmics and x_win_centroid functions to be defined.

    Parameters:
    im (2D array): Graylevel image
    r (float): Radius used for the windowing method

    Returns:
    list of float: x and y coordinates of each centroid"""

    im = del_cosmics(im)
    (x_c_1, y_c_1), win_mat = x_win_centroid(im, r, winMat=True)
    im = del_cosmics((1-win_mat)*im)
    (x_c_2, y_c_2), win_mat = x_win_centroid(im, r, winMat=True)

    return x_c_1, y_c_1, x_c_2, y_c_2

def get_all_cent(a, r, prints=True):
    """Finds all centroid pairs for each frame in a video.
    It requires time and numpy (as np) libraries and frame_cent function to be defined.

    Parameters:
    a (dic): Parsed video using parsing_video function
    r (float): Radius used for the windowing method
    prints (bool): Whether to print progress and running times, 
        expressed as micro seconds per pixel computated

    Returns:
    2D array: Four sublists of a['nframes'] length, two for each
        centroid pair of coordinates with format [[x1s],[y1s],[x2s],[y2s]]"""

    # Find all centroids in each frame
    l = int(a['nframes'])
    pix = 0
    c = np.zeros((4,l))
    t0 = time.time()
    for nf in range(l):
        im = a['frames'][nf]
        c[:,nf] = frame_cent(im, r)
        pix += len(im[0,:])*len(im[:,0])

        if prints: print("Frames computed: {:.0f}/".format(nf)+f'{l}', end='\r')
        
    t1 = time.time()
    tf = 10**6*(t1-t0)/pix
    if prints:
        print(f'{np.round(tf,2)} micros/pixel\t{pix} pixels')
    
    return c

def mean_cent(c, r):
    """Returns position of two mean centroids within a video.
    It requires numpy library as np.

    Parameters:
    c (2D array): All centroids within a video, with format [[x1s],[y1s],[x2s],[y2s]]
    r (float): Radius used for the windowing method

    Returns:
    list of float: x and y coordinates of each mean centroid"""

    all_c = np.zeros((2,np.shape(c)[1]*2))
    all_c[:,:np.shape(c)[1]] = c[:2,:]
    all_c[:,np.shape(c)[1]:] = c[2:,:]
    all_c = all_c.transpose()

    ref1 = all_c[0,:]
    d = 0
    i = 1
    while d < r:
        ref2 = all_c[i,:]
        d = np.sqrt((ref1[0]-ref2[0])**2+(ref1[1]-ref2[1])**2)

        i += 1

    c1 = []
    c2 = []
    for i in range(np.shape(c)[1]*2):
        x, y = all_c[i,:]
        d1 = np.sqrt((ref1[0]-x)**2+(ref1[1]-y)**2)
        d2 = np.sqrt((ref2[0]-x)**2+(ref2[1]-y)**2)
        if d1 < d2:
            c1.append([x, y])
        else:
            c2.append([x, y])
    c1 = np.array(c1)
    c2 = np.array(c2)

    return np.array([[np.mean(c1[:,0]), np.mean(c2[:,0])],
                     [np.mean(c1[:,1]), np.mean(c2[:,1])]])

def t_len(means, centroids):
    """Measures the transversal (perpendicular) length of the centroids in reference to mean centroids.
    It requires numpy library as np.

    Parameters:
    means (list of float): x and y coordinates of each mean centroid
    centroids (list of float): x and y coordinates of each measured centroid

    Returns:
    float: Length of transversal longitude in pixel units"""

    # Line is Ax+By+C=0 -> y=mx+n -> 0=mx-y+n
    A = (means[1,1]-means[1,0])/(means[0,1]-means[0,0]) # = m
    B = -1
    C = means[1,0]-A*means[0,0] # = n
    
    c1_upper = centroids[1,0] > A*centroids[0,0]+C
    c2_upper = centroids[1,1] > A*centroids[0,1]+C
    
    d = []
    for i in range(2):
        x0 = centroids[0,i]
        y0 = centroids[1,i]
        d.append(np.abs(A*x0+B*y0+C)/np.sqrt(A**2+B**2))
    
    if c1_upper == c2_upper:
        return abs(d[0]-d[1])
    else:
        return sum(d)

def l_len(means, centroids):
    """Measures the longitudinal (parallel) length of the centroids in reference to mean centroids.
    It requires numpy library as np.

    Parameters:
    means (list of float): x and y coordinates of each mean centroid
    centroids (list of float): x and y coordinates of each measured centroid

    Returns:
    float: Length of longitudinal length in pixel units"""

    h = np.sqrt((centroids[1,1]-centroids[1,0])**2+(centroids[0,1]-centroids[0,0])**2)
    t_d = t_len(means, centroids)
    
    return np.sqrt(h**2-t_d**2)

def sigma(a, c, resol):
    """Computes the longitudinal and transversal differential variances.
    It requires numpy (as np) library and mean_cent, t_len and l_len functions to be defined.

    Parameters:
    a (dic): Parsed video using parsing_video function
    c (2D array): All centroids within a video, with format [[x1s],[y1s],[x2s],[y2s]]
    resol (float): Resolution of the instrument in arcsec/pixel units

    Returns:
    list of float: longitudinal and transversal differential variances, in arcsec units"""

    r = 20 # Radius used for the windowing method
    means = mean_cent(c, r)

    t_lengths = []
    l_lengths = []
    for i in range(len(c[0,:])):
        centr = np.array([[c[0,i], c[2,i]],
                          [c[1,i], c[3,i]]])
        
        t_lengths.append(t_len(means, centr))
        l_lengths.append(l_len(means, centr))
        
    sig_l = np.std(l_lengths)*resol
    sig_t = np.std(t_lengths)*resol

    return sig_l, sig_t

def fwhm(a, sig_l, sig_t, params):
    """Computes the seeing parameter of a video.
    It requires numpy library as np.

    Parameters:
    a (dic): Parsed video using parsing_video function
    sig_l (float): Longitudinal differential variance, in arcsec units
    sig_t (float): Transversal differential variance, in arcsec units
    params (list of float): Must include (in order) wavelength (in meters),
        sub-aperture diameter (in meters) and distance between sub-apertures (in meters)

    Returns:
    list of float: Altitude correction, longitudinal and transversal seeing and global seeing"""

    w_length, D, B = params
    b = B/D
    cos_z = np.cos(np.radians(90-a['objalt']))
    sig = np.array([sig_l, sig_t])*(np.pi/(3600*180))

    sig2 = sig**2
    K = np.array([0.364*(1-0.532*b**(-1/3)-0.024*b**(-7/3)),
                  0.364*(1-0.798*b**(-1/3)+0.018*b**(-7/3))])

    rad_eps = 0.98*(D/w_length)**0.2*(sig2/K)**0.6
    arcsec_eps = rad_eps*(3600*180/np.pi)

    eps_corr = arcsec_eps*cos_z**(3/5)

    result = cos_z**(3/5), eps_corr[0], eps_corr[1], np.sqrt(eps_corr[0]**2+eps_corr[1]**2)

    return result