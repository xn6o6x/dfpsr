# dfpsr
dedisperse and fold search mode psrfits data

dfpsr.py: 
	Dedisperse and fold the search mode psrfits file with pulsar name (or PSRCAT ephemris file, or DM and period). The folded file is saved as ld format, which is a new format to save pulsar data and information. The data in ld file always have 4 dimensions (nchan, nsub, nbin, npol).

	dfpsr.py [-f FREQ_RANGE] [-d DM] [-p PERIOD] [-n PSR_NAME] [-e PAR_FILE] [-b NBIN] [-a CAL [CAL ...]] [--cal_period CAL_PERIOD] [-s SUBINT] [-m MULTI] filename [filename ...]

ldcomp.py:
	Compress the ld format file with given nchan, nsub, nbin, and save resutls in a new ld file.

	ldcom.py [-f NCHAN] [-F] [-t NSUB] [-T] [-b NBIN] [-B] [-P] [-r FREQ_RANGE] [-s SUBINT_RANGE] filename

ldpara.py:
	View the information of ld format file.

	ldpar.py [-c PARAMETER_NAME_LIST] filename

ldplot.py:
	Plot the time-domain or frequency-domain image or pulse profile of a ld file.

	ldplt.py [-f] [-t] [-p] [-b PHASE_RANGE] [-r FREQ_RANGE] [-s SUBINT_RANGE] filename

ldcal.py:
	Obtain the calibration ld file with periodic noise fits file.

	ldcal.py [--cal_period CAL_PERIOD] filename [filename ...]

ldzap.py:
	Zap the frequency domain interference in ld file.

	ldzap.py filename

lddm.py:
	Calculate the best DM value for ld file.

	lddm.py [-r FREQUENCY] [-s SUBINT] [-n] [-d DM] [-z ZONE] filename

ldconv.py:
	Convert the ld format file to other format.

	ldconv.py [-m MODE] filename

ld.py:
	Provide some functions to access ld format data. With these functions, one can read data and information of ld file, or write data and information in a ld file.

time_eph.py:
	Provide a class to calculate the position of FAST, the ephemris of the solar system, and to convert the time between different time standard.

psr_read.py:
	Provide a class to obtain the ephemris of the pulsar basing on the PSRCAT.

psr_timing.py:
	Provide a class to calculate the pulse phase of a pulsar with a given time and a given frequency.

Dependence: 

	Software: PSRCAT

	Python module: numpy, matplotlib, scipy, astropy, mpmath and tkinter
