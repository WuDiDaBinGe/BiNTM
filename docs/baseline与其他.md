#### 论文中的超参数设置

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: |
|     64     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |

#### 6.28 训练

| epoch | topic num |
| :---: | :-------: |
| 10000 |    20     |

|                           评价指标                           |                           Loss曲线                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![Topic_Coherence](https://gitee.com/yxbLovewy/my-pictures/raw/master/Topic_Coherence.svg) | ![Train_Loss](https://gitee.com/yxbLovewy/my-pictures/raw/master/Train_Loss.svg) |

```python
['israel', 'lebanese', 'lebanon', 'israeli', 'civilian', 'territory', 'arab', 'jews', 'palestinian', 'make']
['driver', 'card', 'gateway', 'latest', 'video', 'run', 'load', 'mouse', 'mode', 'windows']
['jesus', 'sin', 'christian', 'bible', 'christ', 'god', 'love', 'use', 'make', 'lord']
['space', 'nasa', 'billion', 'launch', 'high', 'rocket', 'fly', 'solar', 'moon', 'station']
['simm', 'nec', 'apple', 'keyboard', 'like', 'pin', 'mac', 'suggestion', 'nice', 'advice']
['oil', 'spot', 'bike', 'button', 'air', 'helmet', 'reliable', 'clean', 'plug', 'put']
['church', 'christian', 'bible', 'scientific', 'revelation', 'book', 'part', 'christianity', 'point', 'see']
['game', 'team', 'score', 'cup', 'play', 'ice', 'hockey', 'season', 'make', 'see']
['koresh', 'batf', 'tear', 'compound', 'die', 'fire', 'gas', 'cs', 'agent', 'see']
['car', 'turbo', 'model', 'sport', 'road', 'sit', 'engine', 'handle', 'drive', 'brake']
['shipping', 'original', 'cable', 'offer', 'manual', 'sell', 'floppy', 'please', 'hp', 'new']
['gun', 'criminal', 'firearm', 'violent', 'ban', 'crime', 'make', 'case', 'people', 'see']
['player', 'league', 'play', 'career', 'defensive', 'nhl', 'baseball', 'hockey', 'average', 'roger']
['problem', 'fix', 'compile', 'null', 'patch', 'error', 'bug', 'turn', 'program', 'file']
['armenians', 'armenian', 'turks', 'armenia', 'turkish', 'serve', 'turkey', 'population', 'muslim', 'soviet']
['key', 'algorithm', 'chip', 'clipper', 'secure', 'escrow', 'trust', 'encryption', 'government', 'secret']
['clock', 'slow', 'speed', 'faster', 'cpu', 'mhz', 'fast', 'processor', 'bus', 'handle']
['find', 'hello', 'know', 'please', 'vga', 'file', 'driver', 'program', 'update', 'anyone']
['bike', 'dog', 'dod', 'ride', 'honda', 'make', 'like', 'drink', 'roll', 'disclaimer']
['disease', 'drug', 'much', 'money', 'people', 'tax', 'pay', 'oh', 'take', 'since']

```

| epoch | topic num |
| :---: | :-------: |
| 15000 |    20     |


| 评价指标 | Loss曲线 |
| :---: | :-------: |
| ![](https://gitee.com/yxbLovewy/my-pictures/raw/master/Topic_Coherence_15000.svg) |    ![](/home/yxb/Pictures/Train_Loss_15000.svg)    |

**Batch Size 256时的情况:**Batch size 可能不是影响训练因素

|                           评价指标                           |                           Loss曲线                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20210716091755973](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210716091755973.png) | ![image-20210716091809117](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210716091809117.png) |

![image-20210815183700082](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210815183700082.png)



| ![image-20210815183606819](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210815183606819.png) | ![image-20210815183617948](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210815183617948.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

最好的断点训练的主题：

*ckpt_best_16500.pth*得出的topic-words分布如下：

```shell
['israel', 'israeli', 'arab', 'civilian', 'territory', 'arabs', 'lebanese', 'attack', 'lebanon', 'palestinian']
['driver', 'card', 'mode', 'gateway', 'video', 'program', 'use', 'run', 'load', 'windows']
['sin', 'love', 'christian', 'jesus', 'think', 'day', 'bible', 'god', 'make', 'heaven']
['nasa', 'space', 'launch', 'rocket', 'moon', 'shuttle', 'flight', 'station', 'pat', 'satellite']
['simm', 'apple', 'keyboard', 'nec', 'pin', 'like', 'mac', 'get', 'compatible', 'use']
['oil', 'plug', 'clean', 'air', 'crack', 'helmet', 'make', 'engine', 'spot', 'take']
['science', 'church', 'christian', 'book', 'think', 'bible', 'catholic', 'seem', 'revelation', 'reasoning']
['game', 'team', 'score', 'ice', 'goal', 'win', 'think', 'make', 'season', 'take']
['compound', 'koresh', 'tear', 'batf', 'cs', 'fire', 'gas', 'waco', 'people', 'take']
['car', 'turbo', 'model', 'engine', 'sport', 'speed', 'tire', 'think', 'enough', 'use']
['shipping', 'ship', 'sell', 'email', 'offer', 'please', 'manual', 'original', 'item', 'disk']
['gun', 'criminal', 'firearm', 'crime', 'people', 'violent', 'use', 'ban', 'think', 'see']
['player', 'league', 'nhl', 'play', 'career', 'defensive', 'hockey', 'better', 'average', 'hall']
['problem', 'fix', 'compile', 'error', 'patch', 'client', 'server', 'map', 'null', 'file']
['armenian', 'armenians', 'turkish', 'turks', 'genocide', 'armenia', 'soviet', 'turkey', 'azerbaijan', 'civilian']
['key', 'encryption', 'secure', 'chip', 'escrow', 'clipper', 'trust', 'government', 'crypto', 'algorithm']
['clock', 'slow', 'speed', 'mhz', 'cpu', 'faster', 'processor', 'cache', 'fast', 'bus']
['find', 'please', 'know', 'program', 'anyone', 'look', 'anybody', 'thanks', 'file', 'advance']
['bike', 'dog', 'dod', 'ride', 'honda', 'roll', 'road', 'thing', 'think', 'drink']
['drug', 'money', 'oh', 'disease', 'take', 'people', 'economy', 'crack', 'health', 'tax']
c_a:0.2101156221389105,c_p:0.2774334651257717, npmi:0.050271741293691205
```

**Topic50训练情况：**

| epoch | topic num |
| :---: | :-------: |
| 50000 |    50     |

|                           评价指标                           |                           Loss曲线                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20210706210212173](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210706210212173.png) | ![image-20210706210254300](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210706210254300.png) |

**Topic Num = 30:**

**log:**20news_clean_2021-08-12-23-46_topic30

| ![image-20210813024719462](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210813024719462.png) | ![image-20210813024735042](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210813024735042.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

![image-20210813024617482](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210813024617482.png)



**Topic Num = 50:**

**log:**20news_clean_2021-08-13-02-47_topic50

| ![image-20210813073632779](/home/yxb/.config/Typora/typora-user-images/image-20210813073632779.png) | ![image-20210813073642921](/home/yxb/.config/Typora/typora-user-images/image-20210813073642921.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

![image-20210813073706747](/home/yxb/.config/Typora/typora-user-images/image-20210813073706747.png)

**Topic num=75:**

**log:**atm/20news_clean_2021-08-13-07-38_topic75/

| ![image-20210813145931894](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210813145931894.png) | ![image-20210813145940185](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210813145940185.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

![image-20210813150005976](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210813150005976.png)

**Topic num=100:**

**log:**atm/20news_clean_2021-08-14-11-57_topic100/

| ![image-20210814190930674](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210814190930674.png) | ![image-20210814190945155](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210814190945155.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

![image-20210814190854110](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210814190854110.png)

**NPMI average:**0.058988 

## Contrastive Learning + BTM

| epoch | topic num |   lr      | batch size |
| :---: | :-------: | :-------: | :-------: |
| 50000 |    20    |    1e-4   |    512     |

|                           评价指标                           |                           Loss曲线                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20210709154108009](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210709154108009.png) | ![image-20210709154125595](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210709154125595.png) |

碰到的问题是看出E_loss有明显下降，但是G和D Loss几乎不变，考虑是**学习率**的问题。

Time: 2021.7.9

Train Log:  log/c_atm/20news_clean_2021-07-09-22-19_topic20

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210710101701722](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210710101701722.png) | ![image-20210710101636635](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210710101636635.png) | ![image-20210710101617254](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210710101617254.png) |

这种方法可能不太行

##### Encoder加入了Cluster-Loss

**Time：** 2021-7-11   

**log：**log/c_atm/20news_clean_2021-07-11-21-32_topic20

**Instance temperature： 0.5 cluster temperature = 1**

**记录**:评价指标不太行，Loss_E（红）下降，loss_D深蓝色是上升，loss_G蓝色可能收敛.对比学习的两个Loss收敛的很快。

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210712105608797](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210712105608797.png) | ![image-20210712105720047](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210712105720047.png) | ![image-20210712105704003](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210712105704003.png) |

是不是因为temperature的设置问题？接下来改进temperature的值

**改进了Temperature的值 算Contrastive-Loss时对向量进行L2正则化**

**Time：** 2021-7-12   

**log：**log/c_atm/20news_clean_2021-07-1-21-32_topic20

**Instance temperature：** 0.07  

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210712203410243](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210712203410243.png) | ![image-20210712203454716](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210712203454716.png) | ![image-20210712203444479](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210712203444479.png) |

训练了4000个epoch

------

**改变batch_size = 256,并且为Instance-loss 赋予了Loss权重为0.1**

**Time:**2021-7-12-22-03

**Log:**log/c_atm/20news_clean_2021-07-12-22-03_topic20

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210713174323347](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210713174323347.png) | ![image-20210713174358885](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210713174358885.png) | ![image-20210713174442718](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210713174442718.png) |

------

**使用了新的结构 在Discriminator上加上了project head以及Contrastive loss(Contrastive Loss只加上了对于原始文本的Data Contrastive Loss)**

```python
# 模型结构为：
    def __init__(self, n_topic, v_dim, hid_features_dim, z_features_dim=128):
        super(ContrastiveDiscriminator, self).__init__()
        # doc hidden features
        self.discriminator_encoder = nn.Sequential(
            *block(v_dim, 2048),
            *block(2048, 1024),
            *block(1024, hid_features_dim),
        )
        # doc instance project for contrastive loss
        self.project_head = nn.Sequential(
            nn.Linear(hid_features_dim, hid_features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_features_dim, z_features_dim)
        )
        # doc hidden features + topic
        self.score_head = nn.Sequential(
            *block(n_topic + hid_features_dim, 256),
            nn.Linear(256, 1)
        )

    def forward(self, topic_distribute, doc_bow):
        doc_hidden_features = self.discriminator_encoder(doc_bow)
        contrastive_features = self.project_head(doc_hidden_features)
        p_join = torch.cat([topic_distribute, doc_hidden_features], dim=1)
        score = self.score_head(p_join)
        return score, contrastive_features
```

* Time： 2021.07.14.16.06
* Log：    log/c_atm_discriminator/20news_clean_2021-07-14-16-06_topic20

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210714190120589](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210714190120589.png) | ![image-20210714190137547](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210714190137547.png) | ![image-20210714190155773](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210714190155773.png) |

由Loss可以看出D_loss很快收敛了，而且Contrastive Loss不收敛了上下震荡， 考虑是因为学习率太大的原因，可能是Discriminator结构复杂之后，原来的学习率不适应新的结构了。

**调整了Discriminator的学习率 让他单独使用一个较小的学习率更新**

|                           评价指标                           |                           Loss曲线                           | 对比Loss                                                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![image-20210715151155744](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210715151155744.png) | ![image-20210715151209520](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210715151209520.png) | ![image-20210715151243575](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210715151243575.png) |
| ![image-20210715151447694](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210715151447694.png) | ![image-20210715151459269](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210715151459269.png) | ![image-20210715151507625](https://gitee.com/yxbLovewy/my-pictures/raw/master/mdimgs/image-20210715151507625.png) |

此时的模型 产生了模型坍塌的情况，NPMI的值相比之前的改进强了一点儿

```python
['evil', 'fund', 'somehow', 'original', 'use', 'one', 'get', 'say', '_eos_', 'write']
['evil', 'fund', 'somehow', 'original', 'people', 'believe', 'law', 'government', 'hope', 'go']
['evil', 'fund', 'somehow', 'original', 'people', 'use', 'hope', 'authority', 'year', 'thus']
['evil', 'fund', 'somehow', 'original', 'use', 'one', 'get', 'say', '_eos_', 'write']
['evil', 'fund', 'somehow', 'original', 'hope', 'use', 'get', 'one', '_eos_', 'write']
['evil', 'fund', 'somehow', 'original', 'hope', 'law', 'year', 'speak', 'people', 'police']
```

