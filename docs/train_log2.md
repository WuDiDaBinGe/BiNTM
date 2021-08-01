### 7.20号新方法—在Generator中`加入对比Loss

**Time：**2021-07-20-22-58

**Log：**checkpoint_2021-07-20-22-58_20news_clean_20

**Method**：参考CVPR2021的Text-Image论文以及ContraGAN的思路加入了Generator的Contrastive Loss具体方法如下：

![](https://gitee.com/yxbLovewy/my-pictures/raw/master/discrimitor.png)

![image-20210721105021800](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210721105021800.png)

超参数设置如下：

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |    0.07     |

曲线图：

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210722211234733](/home/yxb/.config/Typora/typora-user-images/image-20210722211234733.png) | ![image-20210722211259462](/home/yxb/.config/Typora/typora-user-images/image-20210722211259462.png) | ![image-20210722211310323](/home/yxb/.config/Typora/typora-user-images/image-20210722211310323.png) |
| ![image-20210721105303104](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210721105303104.png) | ![image-20210721105246141](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210721105246141.png) | ![image-20210721105316642](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210721105316642.png) |

从训练情况看这种方法可以不怎么降低损失，最好的效果也达到了**0.053**左右。



**Time：**2021-07-22-22-18

**Log：**log/gc_atm/20news_clean_2021-07-22-22-18_topic20

**Method**：并且**单独设置了Generator的学习率为：lr/100**，**改变了temperature为0.1**

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.1**   |

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210723143011134](/home/yxb/.config/Typora/typora-user-images/image-20210723143011134.png) | ![image-20210723143051474](/home/yxb/.config/Typora/typora-user-images/image-20210723143051474.png) | ![image-20210723143042676](/home/yxb/.config/Typora/typora-user-images/image-20210723143042676.png) |
| ![image-20210721105303104](/home/yxb/.config/Typora/typora-user-images/image-20210723143025014.png) | ![image-20210721105246141](/home/yxb/.config/Typora/typora-user-images/image-20210723143101362.png) | ![image-20210721105316642](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210721105316642.png) |

效果不错，指标的最好的地方达到了**0.071**

```shell
['car', 'anyone', 'bike', 'new', 'good', 'buy', 'engine', 'oil', 'ride', 'dod']
['game', 'player', 'team', 'play', 'go', 'hockey', 'win', 'nhl', 'score', 'season']
['msg', 'doctor', 'food', 'patient', 'gordon', 'bank', 'disease', 'go', 'treatment', 'surrender']
['window', 'thanks', 'anyone', 'advance', 'display', 'color', 'screen', 'run', 'look', 'appreciate']
['card', 'monitor', 'video', 'mode', 'driver', 'vga', 'color', 'mouse', 'chip', 'speed']
['israel', 'israeli', 'arab', 'jews', 'bank', 'arabs', 'lebanese', 'gordon', 'lebanon', 'research']
['space', 'car', 'launch', 'cost', 'nasa', 'orbit', 'moon', 'shuttle', 'price', 'look']
['article', 'write', 'jewish', 'book', 'anyone', 'michael', 'know', 'name', 'christian', 'read']
['people', 'clinton', 'tax', 'government', 'think', 'go', 'get', 'want', 'batf', 'much']
['god', 'atheist', 'believe', 'people', 'say', 'atheism', 'exist', 'think', 'belief', 'faith']
['turkish', 'armenian', 'armenians', 'armenia', 'turks', 'turkey', 'government', 'law', 'genocide', 'war']
['windows', 'file', 'driver', 'dos', 'font', 'version', 'use', 'program', 'window', 'problem']
['game', 'win', 'year', 'team', 'run', 'hit', 'season', 'last', 'baseball', 'pitch']
['thanks', 'please', 'post', 'file', 'anyone', 'mail', 'email', 'list', 'advance', 'send']
['go', 'car', 'get', 'think', 'bike', 'right', 'back', 'say', 'ride', 'like']
['god', 'jesus', 'christian', 'bible', 'christ', 'sin', 'believe', 'faith', 'say', 'church']
['gun', 'weapon', 'people', 'firearm', 'crime', 'kill', 'criminal', 'death', 'radar', 'amendment']
['drive', 'disk', 'scsi', 'controller', 'ide', 'mac', 'floppy', 'hard', 'modem', 'driver']
['key', 'chip', 'encryption', 'clipper', 'escrow', 'use', 'phone', 'government', 'system', 'algorithm']
['sale', 'offer', 'price', 'sell', 'card', 'condition', 'please', 'shipping', 'buy', 'email']
c_a:0.23019318905854483,c_p:0.3121698165762986, npmi:0.07203471161176403
```

***

**超参数不变，Topic=30的实验结果**

**Time：**2021-07-23-22-28

**Log：**log/gc_atm/20news_clean_2021-07-23-22-28_topic30

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210724111225297](/home/yxb/.config/Typora/typora-user-images/image-20210724111225297.png) | ![image-20210724111246981](/home/yxb/.config/Typora/typora-user-images/image-20210724111246981.png) | ![image-20210724111324656](/home/yxb/.config/Typora/typora-user-images/image-20210724111324656.png) |
| ![image-20210724111357482](/home/yxb/.config/Typora/typora-user-images/image-20210724111357482.png) | ![image-20210724111341895](/home/yxb/.config/Typora/typora-user-images/image-20210724111341895.png) | ![image-20210721105316642](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210721105316642.png) |

***

**超参数不变，Topic=50的实验结果**

**Time：**2021-07-24-11-17

**Log：**log/gc_atm/20news_clean_2021-07-24-11-17_topic50

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210725092602841](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210725092602841.png) | ![image-20210725092847551](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210725092847551.png) | ![image-20210725092626654](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210725092626654.png) |
| ![image-20210724111357482](/home/yxb/.config/Typora/typora-user-images/image-20210724111357482.png) | ![image-20210724111341895](/home/yxb/.config/Typora/typora-user-images/image-20210724111341895.png) | ![image-20210721105316642](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210721105316642.png) |



NPMI值统计：

|   20    |   30    |   50    |   75    |  100   |
| :-----: | :-----: | :-----: | :-----: | :----: |
| 0.07203 | 0.06753 | 0.04457 | 0.03741 | 0.0246 |

**temperature:** 设置较大的值，更容易分开
