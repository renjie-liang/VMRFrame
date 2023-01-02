import  matplotlib.pyplot as plt
a = mask2d.cpu().detach()
plt.matshow(a)
plt.savefig("./figures/tmp.png")



import  matplotlib.pyplot as plt
a = iou2d.cpu()
plt.matshow(a)
plt.savefig("./figures/label2d.png")

