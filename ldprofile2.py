#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import argparse as ap
import os,time,ld,sys
import warnings as wn

#
version='JuiAnHsu_20210408'
#
os.environ["OMP_NUM_THREADS"] = "20" # export OMP_NUM_THREADS=10
os.environ["OPENBLAS_NUM_THREADS"] = "20" # export OPENBLAS_NUM_THREADS=10
os.environ["MKL_NUM_THREADS"] = "20" # export MKL_NUM_THREADS=10
os.environ["VECLIB_MAXIMUM_THREADS"] = "20" # export VECLIB_MAXIMUM_THREADS=10
os.environ["NUMEXPR_NUM_THREADS"] = "20" # export NUMEXPR_NUM_THREADS=10
#
parser=ap.ArgumentParser(prog='ldprofile',description='Extract the profile of the ld file and save to npy file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",help="input ld file")
parser.add_argument('-b','--phase_range',default=0,dest='phase',help='limit the phase range, PHASE0,PHASE1')
parser.add_argument('-r','--frequency_range',default=0,dest='frequency',help='limit the frequency rangeFREQ0,FREQ1')
parser.add_argument('-s','--subint_range',default=0,dest='subint',help='limit the subint range SUBINT0,SUBINT1')
parser.add_argument('--polar',default=0,dest='polar',type=int,help='plot the specified polarization (1234 for IQUV)')
parser.add_argument('-c','--rotation',default=0,dest='rotation',type=np.float64,help='rotate the plot phase')
parser.add_argument('-n',action='store_true',default=False,dest='norm',help='normalized the data at each channel or subint')
args=(parser.parse_args())
wn.filterwarnings('ignore')
#
if not os.path.isfile(args.filename):
	parser.error('A valid ld file name is required.')
d=ld.ld(args.filename)
info=d.read_info()
if info['mode']=='cal':
	parser.error('This ld file is calibration data.')
#
if 'compressed' in info.keys():
	nchan=int(info['nchan_new'])
	nbin=int(info['nbin_new'])
	nsub=int(info['nsub_new'])
	npol=int(info['npol_new'])
else:
	nchan=int(info['nchan'])
	nbin=int(info['nbin'])
	nsub=int(info['nsub'])
	npol=int(info['npol'])
freq_start=np.float64(info['freq_start'])
freq_end=np.float64(info['freq_end'])
freq=(freq_start+freq_end)/2.0
bw=freq_end-freq_start
channel_width=(freq_end-freq_start)/nchan

#
if args.frequency:
	frequency=np.float64(args.frequency.split(','))
	if len(frequency)!=2:
		parser.error('A valid frequency range should be given.')
	if frequency[0]>frequency[1]:
		parser.error("Starting frequency larger than ending frequency.")
	freq_start=max(frequency[0],freq_start)
	freq_end=min(frequency[1],freq_end)
	chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-freq)/channel_width+0.5*nchan))
	chan=np.arange(chanstart,chanend)
	if len(chan)==0:
		parser.error('Input bandwidth is too narrow.')
else:
	frequency=np.array([freq_start,freq_end])
	chan=[]
#
if args.polar:
	polar=args.polar-1
	if polar>npol-1 or polar<0:
		parser.error('The specified polarization is not exist.')
else:
	polar=0
#
if args.subint:
	subint=np.float64(args.subint.split(','))
	if len(subint)!=2:
		parser.error('A valid subint range should be given.')
	if subint[0]>subint[1]:
		parser.error("Starting subint larger than ending subint.")
	subint_start=max(int(subint[0]),0)
	subint_end=min(int(subint[1]+1),nsub)
else:
	subint_start=0
	subint_end=nsub
#
if args.phase:
	phase=np.float64(args.phase.split(','))
	if len(phase)!=2:
		parser.error('A valid phase range should be given.')
	if phase[0]>phase[1]:
		parser.error("Starting phase larger than ending phase.")
else:
	phase=np.array([0,1])
#
def shift(y,x):
	fftp=fft.rfft(y,axis=0)
	ffts=fftp*np.exp(-2*np.pi*x*1j*np.arange(np.shape(fftp)[0])).reshape(-1,1)
	fftr=fft.irfft(ffts,axis=0)
	return fftr

# profile
profile=np.array([])
ordinal=lambda n: "%d%s"%(n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
for start in range(subint_start,subint_end):
	start_time=time.time()
	data=d.chan_scrunch(chan,start,start+1).sum(0)
	bins,mn=10,nbin/10
	stat,val=np.histogram(data[:,0],bins)
	while stat.max()>mn and 2*bins<mn:
		bins*=2
		stat,val=np.histogram(data[:,0],bins)
	val=(val[1:]+val[:-1])/2.0
	if args.rotation:
		data=shift(data,args.rotation)
	x=np.linspace(0,1,len(data))
	if args.polar:
		data=data[:,polar]
	elif npol==4:
		ii,qq,uu,vv=data.T
		profile=np.append(profile,ii)
	else:
		data=data.sum(-1)
	time_mark=time.strftime("%H:%M:%S", time.localtime())
	if args.verbose:
		sys.stdout.write("Extracting %s profile takes %.3f seconds.  %s\n"%(ordinal(start),time.time()-start_time,time_mark))
np.save('.'.join(args.filename.split('.')[:-1])+'_p.npy',profile)
