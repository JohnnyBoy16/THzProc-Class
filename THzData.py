import numpy as np
import os
import copy
import struct
import pdb
import sys
# from THzProc1_12 import ReMap, AmpCor300, FindPeaks
from thz_functions import ReMap, AmpCor300, FindPeaks

# THINGS THAT STILL HAVE TO BE IMPLEMENTED
# Depth Map
# trend off != 0
# AmpCompOn


class THzData:
    # user can provide the full path the file in filename, or provide the filename and base
    # directory separately
    def __init__(self, filename, basedir=None, print_on=True):
        dat = DataFile(filename, basedir=basedir)  # first thing to do is open the file
        self.gate = [[100, 1000], [700, 900]]

        self.x = dat.data[3:]['x']
        self.y = dat.data[3:]['y']
        self.waveform = dat.data['waveform'][3:]
        self.wave_length = len(self.waveform[0])

        self.follow_gate_on = False
        self.a_scan_only = True
        self.b_scan_on = True
        self.b_scan_dir = 'horizontal'  # B Scan direction is usually horizontal by default

        # provide explanation for signal type later, for now just leave it as 1 and look to THz
        # proc for info. Use signal type = 0 to use the front gate regardless of whether follow
        # gate is on or not
        signal_type = 0

        # whether or not to correct for the excessive amplification on the edges of the
        # 300 ps waveforms
        self.amp_correction300_on = True

        # initialize all of the variables that are wanted from the header, these are calculated in
        # the method header_info
        self.time_length = None  # the time length of the scan in ps
        self.x_res = None  # the attempted spacing between x points in the scan in mm
        self.y_res = None  # the attempted spacing between y points in the scan in mm
        self.x_step = None  # the number of steps per row on major axis (usually x-dimension)
        self.y_step = None  # the number of steps per column (usually y-dimension)
        self.x_min = None  # the smallest x value (mm)
        self.y_min = None  # the smallest y value (mm)
        self.x_max = None  # the largest x value (mm)
        self.y_max = None  # the largest y value (mm)
        self.scan_type = None  # the type of scan performed usually '2D Image Scan with Encoder'

        # the first axis, usually x, but can be turntable if rotational scan in performed.
        self.axis = None

        self.delta_t = None  # the spacing between time values
        self.time = None  # array of time values that is used for plotting
        self.freq = None  # array of frequency values that is used for ploting
        self.delta_f = None  # spacing between frequency values
        self.n_half_pulse = None  # the number of half pulses in the scan
        self.true_x_res = None  # the spacing between x points after Remap is called
        self.true_y_res = None  # the spacing between y points after Remap is called
        self.delta_x = None  # the average difference between x data points
        self.delta_y = None  # the average difference between y data points
        self.b_scan = None  # the B-Scan values

        # the largest frequency that you would like the frequency plots to go up to
        self.work_freq = 3.0

        self.follow_gate = None  # Initialize follow gate, but don't give it a value

        # flags peaks (the FSE always and peaks in follow gate if follow gate is on)
        self.flag_peak_on = True

        # CONSTANTS --------------------------------------------------------------------------------

        # 1 for the positive peak, 2 for the negative peak, prefer 1
        self.PICK_PEAK = 1

        # provides algorithm for removing baseline trend in near-field imaging. leave 0 for now
        self.TREND_OFF = 0

        # VERY constant: do not change unless you have a good reason to
        self.FSE_TOLERANCE = -0.17

        # if true, removes excess amplitude on the ends of each waveform for a 300ps scan
        self.AMP_CORRECTION_300 = True
        # parameters to pass to the AmpCor300 function
        self.AMP_CORRECTION_300_PAR = [0., 35., 5.0, 1., 240., 300., 1., 4.5, 4.]

        self.X_CORRECTION_TOLERANCE = 0.1
        self.PULSE_LENGTH = 3.0

        # difference that a point is allowed to deviate (as a ratio with respect to resolution)
        # from its desired coordinate
        self.X_DIFFERENCE_TOLERANCE = 0.4
        self.Y_DIFFERENCE_TOLERANCE = 0.3

        # tolerance ratio for number of scan points in a X line
        # this is nlim from base THzProc
        self.N_LIMIT = 0.8

        # half pulse width for bracketing the follower signal (usually the front surface echo)
        self.HALF_PULSE = 2  # don't change unless you have a good reason to

        # threshold for lead gate signal
        self.FSE_THRESHOLD = 0.5

        # tolerance ratio for number of actual scan points in a X line compared to how many are
        # supposed to be in that line
        self.X_RATIO_LIMIT = 0.8

        # the height of the figure in inches
        self.FIGURE_HEIGHT = 8

        # a small value that is used to see if floating points are close together.
        self.TINY = 1e-4

        # END OF CONSTANTS -------------------------------------------------------------------------

        # Begin driving the methods
        self.header_info(dat.header)
        self.delta_calculator()

        if print_on:
            self.printer()

        # call remap to correct for the backlash of the system
        # the 0 parameter is for XChkTol, which is not used in the function
        self.waveform, self.c_scan, self.x, self.y, self.pos, self.x_step, self.y_step = \
            ReMap(self.waveform, self.x, self.y, self.x_max, self.x_min, self.y_min,
                  self.true_x_res, self.true_y_res, self.x_step, self.y_step, self.scan_type,
                  self.axis, self.wave_length, self.time_length, self.X_CORRECTION_TOLERANCE,
                  self.X_DIFFERENCE_TOLERANCE, self.Y_DIFFERENCE_TOLERANCE, 0, self.TINY,
                  self.TREND_OFF, self.N_LIMIT)

        # run a check to make sure that the time length is actually 300
        if self.amp_correction300_on and abs(300 - self.wave_length) < self.TINY:
            AmpCor300(self.AMP_CORRECTION_300_PAR[0], self.AMP_CORRECTION_300_PAR[1],
                      self.AMP_CORRECTION_300_PAR[2], self.AMP_CORRECTION_300_PAR[3],
                      self.AMP_CORRECTION_300_PAR[4], self.AMP_CORRECTION_300_PAR[5],
                      self.AMP_CORRECTION_300_PAR[6], self.AMP_CORRECTION_300_PAR[7],
                      self.AMP_CORRECTION_300_PAR[8], self.wave_length, self.delta_t, self.x_step,
                      self.y_step, self.waveform)

        if self.follow_gate_on:
            self.bin_range = copy.deepcopy(self.gate)
            self.peak_bin = np.zeros((5, 2, self.x_step, self.y_step))
        elif not self.follow_gate_on:
            self.bin_range = [[0, self.wave_length]]
            self.peak_bin = np.zeros((5, 1, self.x_step, self.y_step))
        else:
            raise ValueError('follow_gate_on must be set to either True or False')

        # Return peak_bin to keep track of index values for various points on the waveform
        # peak_bin is 5 dimensional
        # index 0 = positive peak
        # index 1 = negative peak
        # index 2 = half way between positive and negative peak
        # index 3 = left gate index
        # index 4 = right gate index
        if self.follow_gate_on:
            self.find_peaks()

        self.c_scan_extent = (self.x_min, self.x_max, self.y_max, self.y_min)
        self.b_scan_extent = (self.x_min, self.x_max, self.time_length, 0)

        # Generate the C-Scan
        self.make_c_scan(signal_type)

    def make_c_scan(self, signal_type=0):
        """
        Generates the C-Scan based on the gate location and signal type
        At the moment this just does peak to peak voltage response
        :param: signal_type - determines how the C-Scan is calculated
                    choices:
                        0: (default) Use Peak to Peak voltage with the front gates regardless of
                           whether follow gate is on or not.
                        1: Use Peak to Peak voltage with the follow gates if on. If follow gate
                           is not on then use peak to peak within front gate
                        2: avg mag (mag = abs amp)
                        3: median mag
                        4: min mag
                        5: max mag
                        6: avg amp ~ integrated gate
                        7: median amp
                        8: min amp
                        9: max amp
        """
        # TODO: implement sig_type
        # TODO: work on vectorizing code to avoid the for loops
        # use Vpp within the front gates regardless of whether follow gate is on or not
        # the gates are ABSOLUTE INDEX POSITION and DO NOT account for differences in height of the
        # front surface
        if signal_type == 0:
            max_amp = np.amax(self.waveform[:, :, self.gate[0][0]:self.gate[0][1]], axis=2)
            min_amp = np.amin(self.waveform[:, :, self.gate[0][0]:self.gate[0][1]], axis=2)
            self.c_scan = max_amp - min_amp

        # use Vpp within the follow gates if on, else within front gates
        # It appears that using peak bin does not allow for vectorization
        elif signal_type == 1:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    max_amp = self.waveform[i, j, self.peak_bin[0, self.follow_gate_on, i, j]]
                    min_amp = self.waveform[i, j, self.peak_bin[1, self.follow_gate_on, i, j]]
                    self.c_scan[i, j] = max_amp - min_amp

        # use avg mag within the gate (abs amplitude within the gate, sum up then avg)
        elif signal_type == 2:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.sum(np.abs(self.waveform[i, j, L:R])) / (R - L)

        # median magnitude within the gate
        elif signal_type == 3:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.median(np.abs(self.waveform[i, j, L:R]))

        # min magnitude within the gate
        elif signal_type == 4:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.min(np.abs(self.waveform[i, j, L:R]))

        # max magnitude within the gate
        elif signal_type == 5:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.amax(np.abs(self.waveform[i, j, L:R]))

        # avg amp within the gate ~ integrated gate
        elif signal_type == 6:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.sum(self.waveform[i, j, L:R]) / (R - L)

        # median amp within the gate
        elif signal_type == 7:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.median(self.waveform[i, j, L:R])

        # min amp within the gate
        elif signal_type == 8:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.min(self.waveform[i, j, L:R])

        # max amp within the gate
        elif signal_type == 9:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    L = self.peak_bin[3, self.follow_gate_on, i, j]
                    R = self.peak_bin[4, self.follow_gate_on, i, j]
                    self.c_scan[i, j] = np.amax(self.waveform[i, j, L:R])

        # 27JAN2015 FSE handle: if FSE too small, throw away this location
        if signal_type != 0 and self.FSE_TOLERANCE > 0:
            for i in range(self.y_step):
                for j in range(self.x_step):
                    if (self.waveform[i, j, self.peak_bin[0, 0, i, j]] -
                            self.waveform[i, j, self.peak_bin[1, 0, i, j]] < self.FSE_TOLERANCE):
                        self.c_scan[i, j] = 0.

    def make_b_scan(self, yid, xid):
        """
        Generates the B-Scan based on the last cursor location clicked and whether b_scan_dir is
        horizontal or vertical
        :param yid: the row from which to generate B-Scan
        :param xid: the column clicked from which to generate B-Scan
        """
        # in order to vectorize this method, the x or y coordinates are put in the rows, but to plot
        # we want them in the columns. Thus the call to transpose after arranging the data
        if self.b_scan_dir == 'horizontal':
            self.b_scan = self.waveform[yid, :, :]
        else:  # b_scan_dir == 'vertical'
            self.b_scan = self.waveform[:, xid, :]

        # call to transpose is necessary to flip data axis, so x or y location is on the bottom
        # time is along the y-axis on the plot
        self.b_scan = np.transpose(self.b_scan)

    def set_b_scan_on(self, given_boolean):
        pass
        # TODO: write code so setting B-Scan on when it was previously off,
        # TODO: calls make B-Scan from the last point clicked
        if self.b_scan_on is given_boolean:
            return  # do nothing

        self.b_scan_on = given_boolean
        if self.b_scan_on:
            self.make_b_scan()

    def set_b_scan_direction(self, given_direction):
        pass
        # TODO: write code so changing B-Scan direction will call make B-Scan from last point clicked
        orientation = given_direction.lower()  # convert given_direction to lower case
        if orientation == 'vertical' or 'horizontal':
            self.b_scan_dir = given_direction
            self.make_b_scan()
        else:
            raise ValueError('B-Scan direction must be either "vertical" or "horizontal"')

    def find_peaks(self):
        """
        Put FindPeaks in method so it can be called in another program. Initializes peak_bin,
        which contains information about the location of the front surface echo, and if follow
        gate is on, the location of the 2nd peak.
        """
        self.peak_bin = FindPeaks(self.waveform, self.x_step, self.y_step, self.wave_length,
                                  self.n_half_pulse, self.FSE_THRESHOLD, self.bin_range,
                                  self.follow_gate_on)

    def printer(self):
        """
        Prints information about the scan to the console when a THz data class in instantiated
        """
        print
        print ' asn wave length =', self.wave_length, ' asn Time Length =', self.time_length, \
            ' delta_t =', self.delta_t, ' delta_f =', self.delta_f, ' scan type =', self.scan_type

        print 'X min =', self.x_min, ' max =', self.x_max, 'Y min =', self.y_min, ' max=', \
            self.y_max, ' scan step  X =', self.x_step, ' Y =', self.y_step, ' res X =', \
            self.x_res, ' Y =', self.y_res

        print 'True resolution:  X =', self.true_x_res, ' Y =', self.true_y_res
        print

    def header_info(self, header):
        """
        Retrieves information about the scan from the header file that is created by the DataFile
        class.
        :param header: the header of the data file class
        """
        # get key parameters from the header
        # last update: 17FEB2013

        for item in header:
            b = item.strip()
            a = b.split()
            if a[0] == 'RSDLAMP:':
                self.time_length = float(a[1])
            elif a[0] == 'RESOLUTION:':
                self.x_res = float(a[1])
            elif a[0] == 'Y_RESOLUTION:':
                self.y_res = float(a[1])
            elif a[0] == 'XSTEPS:':
                self.x_step = int(float(a[1]))  # can't directly int(a[1]) of string a[1] ?!
            elif a[0] == 'YSTEPS:':
                self.y_step = int(float(a[1]))
            elif a[0] == 'XMIN:':
                self.x_min = float(a[1])
            elif a[0] == 'XMAX:':
                self.x_max = float(a[1])
            elif a[0] == 'YMIN:':
                self.y_min = float(a[1])
            elif a[0] == 'YMAX:':
                self.y_max = float(a[1])
            elif a[0] == 'SCAN_NAME:':
                self.scan_type = b[10:].strip()
            elif a[0] == 'AXIS1:':
                self.axis = b[7:].strip()

    def delta_calculator(self):
        """
        Calculates delta_t, delta_f, n_half_pulse, and the time and frequency array that are used
        for plotting. Note that len(freq) is wave_length/2 + 1. This is so it can be used with
        numpy's rfft function. Thomas's THzProc code uses len(freq) as wave_length/2.
        """
        self.delta_t = self.time_length / (self.wave_length - 1)
        self.delta_f = 1. / (self.wave_length * self.delta_t)
        self.time = np.linspace(0., self.time_length, self.wave_length)
        self.freq = np.linspace(0., (self.wave_length/2) * self.delta_f, self.wave_length/2+1)
        self.n_half_pulse = int(self.HALF_PULSE / self.delta_t)
        self.true_x_res = (self.x_max - self.x_min) / float(self.x_step - 1)
        self.true_y_res = (self.y_max - self.y_min) / float(self.y_step - 1)
        self.delta_x = (self.x_max - self.x_min) / float(self.x_step)
        self.delta_y = (self.y_max - self.y_min) / float(self.y_step)


class DataFile(object):
    def __init__(self, filename, basedir=None):
        if basedir is not None:  # allows the user to pass through the full filename as 1 argument
            filename = os.path.join(basedir, filename)
        with open(filename, 'rb') as fobj:
            self.header = list(iter(fobj.readline, "ENDOFHEADER\r\n"))
            # the first 4-byte word after the header gives the length of each row
            first_word = fobj.read(4)
            # convert the 4-byte string to a float
            col_size = struct.unpack(">f", first_word)[0]
            # move the read point back so we can read the first row in it's entirety
            fobj.seek(-4, 1)

            # define a compound data type for the data: the two coordinate values
            # followed by the THz waveform
            bin_dtype = [("x", ">f"), ("y", ">f"), ("waveform", ">f", col_size - 2)]

            # read the data into an array
            self.data = np.fromfile(fobj, bin_dtype)
