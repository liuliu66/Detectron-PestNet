#example
import pickle as p
import matplotlib.pyplot as plt
import pylab as pl
f = open('10000_pr.pkl','rb')
info = p.load(f)
print(info['rec'].shape)
print(info['prec'])
print(info['prec'].shape)
print(info['ap'])
pl.plot(info['rec'], info['prec'], lw=2)
pl.xlabel('Recall')
pl.ylabel('Precision')
#plt.grid(True)
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall')
pl.legend(loc="upper right")     
plt.show()
