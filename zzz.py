import  matplotlib.pyplot as plt
a = label1d_model1s[:,0,0].cpu().detach()
# plt.matshow(a)
plt.plot(a)
plt.savefig("./figures/s_label1d.png")

import  matplotlib.pyplot as plt
plt.cla()
a = label1ds[0, 0, :].cpu().detach()
plt.plot(a)
plt.savefig("./figures/label1d_s.png")
plt.cla()
a = label1ds[0, 1, :].cpu().detach()
plt.plot(a)
plt.savefig("./figures/label1d_e.png")


plt.cla()
a = label1d_model1s[0, 0, :].cpu().detach()
plt.plot(a)
plt.savefig("./figures/label1d_SeqPAN_s.png")
plt.cla()
a = label1d_model1s[0, 1, :].cpu().detach()
plt.plot(a)
plt.savefig("./figures/label1d_SeqPAN_e.png")



import  matplotlib.pyplot as plt
a = labels2d[0].cpu().detach()
plt.matshow(a)
plt.savefig("./figures/labels2d.png")

import  matplotlib.pyplot as plt
plt.cla()
a = scores2d[0].cpu().detach()
plt.matshow(a)
plt.savefig("./figures/scores2d.png")



import  matplotlib.pyplot as plt
a = iou2d.cpu()
plt.matshow(a)
plt.savefig("./figures/label2d.png")


