import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import re
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 0:name
# 1:rowA
# 2:colA
# 3:nnzA
# 4:dasp_pre_time
# 5:dasp_spmv_time
# 6:dasp_spmv_gflops    
# 7:cusparse_pre_time
# 8:cusparse_spmv_time
# 9:cusparse_spmv_gflops

filename = '../data/a100_f16_record.csv'
fn_split = re.split('[/_.]+', filename)
figname = fn_split[2] + '_' + fn_split[3] + '_bar.pdf'

labels = ['dc2', 'scir...', 'mac_...', 'webb...', 'ASIC...', 'Full...', 'rma1...', 
          'eu-2...', 'in-2...', 'cant', 'circ...', 'cop2...', 'mc2d...', 'pdb1...', 
          'conf...','cons...', 'ship...', 'Si41...', 'mip1', 'pwtk','Ga41...']
matrixname = ['dc2.mtx', 'scircuit.mtx', 'mac_econ_fwd500.mtx', 'webbase-1M.mtx', 'ASIC_680k.mtx', 'FullChip.mtx', 'rma10.mtx', 
          'eu-2005.mtx', 'in-2004.mtx', 'cant.mtx', 'circuit5M.mtx', 'cop20k_A.mtx', 'mc2depi.mtx', 'pdb1HYS.mtx', 
          'conf5_4-8x8-10.mtx','consph.mtx', 'shipsec1.mtx', 'Si41Ge41H72.mtx', 'mip1.mtx', 'pwtk.mtx','Ga41As41H72.mtx']

sizeofr = 21
sizeofc = 21

content1 = [[0.0 for i in range(sizeofc)] for i in range(sizeofr)]


font3 = {'family' : 'Liberation Sans',
'weight' : 'normal',
'size'   : 18,}

total_line = len(open(filename).readlines())
with open(filename,'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for num in range(total_line):
        for i in range(sizeofr):
            if matrixname[i] in rows[num][0]:
                content1[i] = rows[num]
                  
data_dasp = [0.0 for i in range(sizeofr)]
data_cusp = [0.0 for i in range(sizeofr)]


for num in range(sizeofr):
    data_dasp[num] = (float)(content1[num][6])
    data_cusp[num] = (float)(content1[num][9])
    
    

x = np.arange(len(labels))  # the label locations
# y = np.arange(len(men_means))
width = 0.17  # the width of the bars

fig,ax = plt.subplots(figsize=(36, 5))

plt.bar(x - width, data_cusp, width,color='#f6b654',edgecolor='black',linewidth=1.5, label='cuSPARSE (v12.0)')
plt.bar(x, data_dasp, width,color='#ee6a5b',edgecolor='black',linewidth=1.5, label='DASP (this work)')

for a,b in zip(x,data_cusp): 
    plt.text(a-width,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)

for a,b in zip(x,data_dasp): 
    plt.text(a,b+2,'%.2f' %b,ha = 'center',va = 'bottom',rotation =90,fontsize=20)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Performance (Gflops)',fontsize=24)
plt.ylim(0,600)

ax.set_xticks(x)
ax.set_xticklabels(labels,rotation=0, fontsize = 22)
plt.tick_params(labelsize=22)

plt.grid(c='grey',alpha=0.8,linestyle='--')

fig.legend(bbox_to_anchor=(0.28, 0.87), loc=1, borderaxespad=0,fontsize=22,ncol=2)


fig.tight_layout()
plt.savefig(figname,dpi=300)
# plt.show()




