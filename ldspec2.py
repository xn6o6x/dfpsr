import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import argparse as ap
import ld,os,copy,time
#
version='JuiAnHsu_20210107'
parser=ap.ArgumentParser(prog='ldspec',description='Save the frequency spectrum as npy file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",help="input ld file")
args=(parser.parse_args())
#
timemark=time.time()
#
if not os.path.isfile(args.filename):
	parser.error('A valid ld file name is required.')
d=ld.ld(args.filename)
info=d.read_info()
#
if 'compressed' in info.keys():
	nchan=int(info['nchan_new'])
	nbin=int(info['nbin_new'])
	nperiod=int(info['nsub_new'])
else:
	nchan=int(info['nchan'])
	if info['mode']=='test':
		nbin=1
		nperiod=int(d.read_shape()[1])
	else:
		nbin=int(info['nbin'])
		nperiod=int(info['nsub'])
npol=int(info['npol'])
if nbin!=1:
	data=d.period_scrunch()[:,:,0]
else:
	data=d.__read_bin_segment__(0,nperiod)[:,:,0]
if nbin>128 or ((nbin==1)&(nperiod>128)):
	data=fft.irfft(fft.rfft(data,axis=1)[:,:65],axis=1)
testdata=copy.deepcopy(data)
testdata=ma.masked_where(testdata<0,testdata)
if 'zchan' in info.keys():
	zaplist=[map(int,info['zchan'].split(','))]
	zapnum=zaplist[0]
	zaparray=np.zeros_like(testdata)
	zaparray[zapnum,:]=True
	testdata.mask=zaparray
	zap0=1
else:
	zaplist=[]
	zap0=0
#
spec=testdata.sum(1)
spec=spec-np.min(spec)
spec0=np.append(0,np.append(spec.repeat(2),0))
spec1=copy.deepcopy(spec0)
freq_start,freq_end=np.float64(info['freq_start']),np.float64(info['freq_end'])
ylim0=[freq_start,freq_end]
channelwidth=(freq_end-freq_start)/nchan
halfwidth=channelwidth/2
freq=np.linspace(ylim0[0]-halfwidth,ylim0[1]-halfwidth,len(spec)+1).repeat(2)
#
np.save('.'.join(args.filename.split('.')[:-1])+'_spec.npy',spec1.compressed())
#
if args.verbose:
	sys.stdout.write('Extracting the spectrum takes '+str(time.time()-timemark)+' second.\n')
