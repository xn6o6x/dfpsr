import ld
import os
import argparse as ap

parser=ap.ArgumentParser(prog='Check_phase0',description='Check the number of the first period.')
parser.add_argument("filename",nargs='+',help="name of file")
args=(parser.parse_args())
d=ld.ld(os.getcwd()+"/"+args.filename[0])
info=d.read_info()
print(info["phase0"])
