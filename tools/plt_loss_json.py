#!/usr/bin/env python
import matplotlib.pyplot as plt
import re
# read the log file
fp = open('nohup.out', 'r')

train_iterations = []
train_loss = []
bbox_loss = []
cls_loss = []
rpn_cls_loss = []
rpn_box_loss = []

for ln in fp:
    # get train_iterations and train_loss
    if 'json_stats' in ln:
        #print(ln)
        iters = re.findall(r'"iter": \d+', ln)
        loss = re.findall(r'"loss": \d+.\d+', ln)
        #print(loss)
        #print(int(arr[0].strip(',')[8:]))
        train_iterations.append(int(iters[0].strip(',')[8:]))
        train_loss.append(float(loss[0].strip(',')[8:]))
        #train_loss.append(float(ln.strip().split(' = ')[-1]))
    if 'output #0: loss_bbox' in ln:
        arr = re.findall(r'loss_bbox = \d+.\d+[e,-]*\d+', ln)
        bbox_loss.append(float(arr[0].split(' ')[-1]))
    if 'output #1: loss_cls' in ln:
        arr = re.findall(r'loss_cls = \d+.\d+[e,-]*\d+', ln)
        cls_loss.append(float(arr[0].split(' ')[-1]))
    if 'output #2: rpn_cls_loss' in ln:
        arr = re.findall(r'rpn_cls_loss = \d+.\d+[e,-]*\d+', ln)
        rpn_cls_loss.append(float(arr[0].split(' ')[-1]))
    if 'output #3: rpn_loss_bbox' in ln:
        arr = re.findall(r'rpn_loss_bbox = \d+.\d+[e,-]*\d+', ln)
        rpn_box_loss.append(float(arr[0].split(' ')[-1]))

fp.close()

#print(train_iterations)
#print(train_loss)
#new added
train_loss_new = []
train_iterations_new = []
bbox_loss_new = []
cls_loss_new = []
rpn_cls_loss_new = []
rpn_box_loss_new = []
num = 10
for i in range(int(len(train_loss)/num)-1):
    train_loss_new.append(sum(train_loss[num*i:num*(i+1)])/num)
    bbox_loss_new.append(sum(train_loss[num*i:num*(i+1)])/num)
    cls_loss_new.append(sum(train_loss[num*i:num*(i+1)])/num)
    rpn_cls_loss_new.append(sum(train_loss[num*i:num*(i+1)])/num)
    rpn_box_loss_new.append(sum(train_loss[num*i:num*(i+1)])/num)
    train_iterations_new.append((train_iterations[num*i] + (train_iterations[num*(i+1)]-train_iterations[num*i])/2))

#print (rpn_cls_loss.index(max(rpn_cls_loss)))
#print (rpn_cls_loss[rpn_cls_loss.index(max(rpn_cls_loss))])
fig = plt.figure()
plt.plot(train_iterations_new, train_loss_new, 'k', label="all loss")
plt.xlim([0, max(train_iterations)])
#plt.ylim([0, 3])
#plt.xlabel('iteration')
#plt.ylabel('loss')
#plt.title('all loss')
#plt.savefig('all loss curve_FPN')

fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 2, 1)
ax2 = fig2.add_subplot(2, 2, 2)
ax3 = fig2.add_subplot(2, 2, 3)
ax4 = fig2.add_subplot(2, 2, 4)

# plot curves
ax1.plot(train_iterations_new, bbox_loss_new, 'k', label="bbox loss")
ax1.set_xlim([0, max(train_iterations)])
#ax1.set_ylim([0, 1])
ax1.set_xlabel('iterations')
ax1.set_ylabel('bbox_loss')
ax1.legend(loc='best')
ax1.set_title('bbox loss curve')

ax2.plot(train_iterations_new, cls_loss_new, 'k', label="cls loss")
ax2.set_xlim([0, max(train_iterations)])
#ax1.set_ylim([0, 1])
ax2.set_xlabel('iterations')
ax2.set_ylabel('bbox_loss')
ax2.legend(loc='best')
ax2.set_title('cls loss curve')

ax3.plot(train_iterations_new, rpn_cls_loss_new, 'k', label="rpn cls loss")
ax3.set_xlim([0, max(train_iterations)])
#ax3.set_ylim([0, 1])
ax3.set_xlabel('iterations')
ax3.set_ylabel('bbox_loss')
ax3.legend(loc='best')
ax3.set_title('rpn cls loss curve')

ax4.plot(train_iterations_new, rpn_box_loss_new, 'k', label="rpn box loss")
ax4.set_xlim([0, max(train_iterations)])
#ax4.set_ylim([0, 1])
ax4.set_xlabel('iterations')
ax4.set_ylabel('bbox_loss')
ax4.legend(loc='best')
ax4.set_title('rpn box loss curve')
plt.subplots_adjust(wspace=0.4, hspace=0.4)
#plt.savefig('4 part loss curve')
plt.draw()
plt.show()