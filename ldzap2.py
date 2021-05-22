#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import argparse as ap
import ld,os,copy,sys,time
#
version='JuiAnHsu_202105'
parser=ap.ArgumentParser(prog='ldzap2',description='Zap the frequency domain with given zap_file.txt and save to ld file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("-z","--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",help="input ld file")
args=(parser.parse_args())
#
os.environ["OMP_NUM_THREADS"] = "20" # export OMP_NUM_THREADS=10
os.environ["OPENBLAS_NUM_THREADS"] = "20" # export OPENBLAS_NUM_THREADS=10
os.environ["MKL_NUM_THREADS"] = "20" # export MKL_NUM_THREADS=10
os.environ["VECLIB_MAXIMUM_THREADS"] = "20" # export VECLIB_MAXIMUM_THREADS=10
os.environ["NUMEXPR_NUM_THREADS"] = "20" # export NUMEXPR_NUM_THREADS=10
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
if args.zap_file:
    if not os.path.isfile(args.zap_file):
        parser.error('The zap channel file is invalid.')
    zchan=np.loadtxt(args.zap_file,dtype=np.int32)
    if np.max(zchan)>=nchan or np.min(zchan)<0:
        parser.error('The zapped channel number is overrange.')
    zap0+=1
    zaplist.append(zchan)
    zapnum=set()
    for i in zaplist:
        zapnum.update(i)
    zapnum=np.array(list(zapnum))
    zaparray=np.zeros_like(testdata)
    zaparray[zapnum,:]=True
    testdata.mask=zaparray
#
zapnum=set()
for i in zaplist:
    zapnum.update(i)
zapnum=np.sort(list(zapnum))
zapnum=list(zapnum[(zapnum>=0)&(zapnum<nchan)])
info['zchan']=str(zapnum)[1:-1]
save=ld.ld('.'.join(args.filename.split('.')[:-1])+'_zap.ld')
save.write_shape([nchan,nperiod,nbin,npol])
ordinal=lambda n: "%d%s"%(n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
for i in np.arange(nchan):
    time_mark=time.strftime("%H:%M:%S", time.localtime())
    if args.verbose:
        sys.stdout.write("Zapping %s channel... %s\n"%(ordinal(i),time_mark))
    if i in zapnum:
        save.write_chan(np.zeros(nperiod*nbin*npol),i)
        continue
    save.write_chan(d.read_chan(i),i)
save.write_info(info)
