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

新模型：

1.Contrastive Loss肯定是有用的

2.要不要在Discriminator上加入

20NewsGroup labels:

![image-20210723104544904](/home/yxb/.config/Typora/typora-user-images/image-20210723104544904.png)

| comp.graphics comp.sys.ibm.pc.hardware comp.sys.mac.hardware comp.windows.x<br />comp.os.ms-windows.misc | rec.autos rec.motorcycles rec.sport.baseball rec.sport.hockey | sci.crypt sci.electronics sci.med sci.space           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- |
| misc.forsale                                                 | talk.politics.misc talk.politics.guns talk.politics.mideast  | talk.religion.misc alt.atheism soc.religion.christian |

***

#### 评价指标

![image-20210725103453090](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210725103453090.png)

![image-20210725103436857](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210725103436857.png)

![image-20210725103525483](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210725103525483.png)

C_p：本方法也是基于滑动窗口，但分词方法为 one-preceding（每个词只与位于其前面和后面的词组成词对），并利用 Fitelson 相关度来表征连贯度。

C_v (Coefficient of variance)：本方法基于滑动窗口，对主题词进行 one-set 分割（一个 set 内的任意两个词组成词对进行对比），并使用归一化点态互信息 (NPMI) 和余弦相似度来间接获得连贯度

***

#### 关于加快训练

三个服务器时间437.88712978363037

一个服务器时间441.4029595851898

分开测试427.9044966697693

分开测试用一个418.9143579006195


接到V维向量后效果较差。
