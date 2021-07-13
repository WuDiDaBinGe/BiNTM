1. 为什么W_GAN中不建议用Adam优化器？而在Paper工作中使用了Adam(BiGan是用的Adam)

##### 2. Data Augment

**TF-IDF word replacing**:

![image-20210706211703428](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210706211703428.png)

##### 3. Batch Size and Learning rate

##### 4. 加到Encoder效果不好

思考：

1.  将对比学习部分加入到generator的部分？

2. 结合Cluster-Instace Loss;Encoder使用Instance contrastive-loss，而Generator使用Cluter-Instance Loss.

   Encorder+Instance和Cluster Loss效果不好

3. 使用不同的Data argument的方法？可以参考**SCCL**论文中的方法

4. 对比学习对向量进行**L2正则化**.

​			
