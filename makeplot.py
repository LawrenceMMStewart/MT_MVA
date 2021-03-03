import os 
import json
import matplotlib.pyplot as plt
import glob
import argparse


parser = argparse.ArgumentParser(description='Plot experiment')
parser.add_argument('prefix',type = str, help = 'prefix of exp to plot')
parser.add_argument('vals',type = str , help = 'values to plot from all files found in order')
parser.add_argument('-l','--legend', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--save',type=str,default = "" ,
	help='save name , if left blank then dont save.')
parser.add_argument('--logplot',type=bool,default=False,help = 'logplot or standard')
args = parser.parse_args()


relevant_repos = glob.glob(f'results/{args.prefix}*')

all_vals = []

for file in relevant_repos:
	with open(os.path.join(file,'params.json')) as f:
		params = json.load(f)
	all_vals.append((params['exp_name'],params[args.vals]))




assert len(all_vals) == len(args.legend)
plt.style.use('ggplot')
plt.plot()
plt.grid('on')
for i in range(len(all_vals)):
	vals = all_vals[i][1]
	label = args.legend[i]
	xvals = [i+1 for i in range(len(vals))]
	plt.plot(xvals,vals,alpha=0.8,label = label)
if args.logplot:
	plt.semilogy()

if args.vals in ['epps','tpps']:
	ytag = 'Perplexity'
elif args.vals in ['elosses','tlosses']:
	ytag = 'Loss'
elif args.vals in ['eacss','taccs']:
	ytag = 'Accuracy'

plt.ylabel(ytag)
plt.xlabel('Epoch')
plt.legend()
if args.save !="":
	plt.savefig(args.save)

plt.show()