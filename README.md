# MT-Food-Classification

for course project

将MT-1000中的train,val,test文件夹放入data中

目前用的是resnet-101做为图片的特征提取， 然后用一层FC跟softmax来进行分类，直接调用torch.topk来取得前三，之后要修改应该主要专心在model.py上就好了，我把
提取预训练特征的网络跟分类用的网络分成两个class，避免两个互相冲突到。

输出的CSV会存在data文件夹中
