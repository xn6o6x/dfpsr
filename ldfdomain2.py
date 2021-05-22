#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import argparse as ap
import os,time,ld,sys
import warnings as wn
#
version='JuiAnHsu_20210414'
parser=ap.ArgumentParser(prog='ldfdomain',description='Save the fdomain data in a 3D array.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",help="input ld file")
parser.add_argument('-b','--phase_range',default=0,dest='phase',help='limit the phase range, PHASE0,PHASE1')
parser.add_argument('-r','--frequency_range',default=0,dest='frequency',help='limit the frequency rangeFREQ0,FREQ1')
parser.add_argument('-s','--subint_range',default=0,dest='subint',help='limit the subint range SUBINT0,SUBINT1')
parser.add_argument('-o','--polynomial_order',default=0,dest='n',type=int,help='fit the back ground with Nth order polynomial')
parser.add_argument('--polar',default=0,dest='polar',type=int,help='plot the specified polarization (1234 for IQUV)')
parser.add_argument('-c','--rotation',default=0,dest='rotation',type=np.float64,help='rotate the plot phase')
parser.add_argument('-n',action='store_true',default=False,dest='norm',help='normalized the data at each channel or subint')
parser.add_argument('-i',action='store_false',default=True,dest='title',help='hide file information above the figure')
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
	subint=np.array([subint_start,subint_end])
#
data=d.period_scrunch(subint_start,subint_end,chan)[:,:,polar]
if 'zchan' in info.keys():
    if len(chan):
        zchan=np.array(list(set(np.int32(info['zchan'].split(','))).intersection(chan)))-chanstart
    else:
        zchan=np.int32(info['zchan'].split(','))
    zaparray=np.zeros_like(data)
    zaparray[zchan]=True
    data=ma.masked_array(data,mask=zaparray)
if args.n:
    data-=np.polyval(np.polyfit(np.arange(nbin),data.T,args.n),np.array([range(nbin)]*len(data)).T).T
else:
    data-=data.mean(1).reshape(-1,1)
if args.norm:
    data/=data.max(1).reshape(-1,1)
if args.rotation:
    data=shift(data,args.rotation)
sys.stdout.write('Subint %s completed.\n'%subint_start)
np.save('.'.join(args.filename.split('.')[:-1])+'_f_%s.npy'%subint_start,data.data[::-1])
