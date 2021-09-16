### InfoNCELoss损失实验记录

**Time：**2021-08-18-23-12

**Log：**gc_atm/20news_clean_2021-08-18-23-12_topic100/

**Method**：并且**单独设置了Generator的学习率为：lr/100**，**改变了temperature为0.1**

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.1**   |

![image-20210819105105019](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210819105105019.png)

---

**Time：**2021-08-19-10-54

**Log：**20news_clean_2021-08-19-10-54_topic100

**Method**：并且**单独设置了Generator的学习率为：lr**，**改变了temperature为0.5**

**Topic Num:**100

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210819192642153](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210819192642153.png)

---

**Time：**021-08-19-23-12

**Log：**gc_atm/20news_clean_2021-08-19-23-12_topic75/

**Method**：并且**单独设置了Generator的学习率为：lr**，**改变了temperature为0.5**

**Topic Num:**75

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210820095317332](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210820095317332.png)

---

**Time：**20news_clean_2021-08-20-10-08_topic50

**Log：**gc_atm/20news_clean_2021-08-20-10-08_topic50

**Method**：并且**单独设置了Generator的学习率为：lr**，**改变了temperature为0.5**

**Topic Num:**50

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210820213432499](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210820213432499.png)

---

**Time：**20news_clean_2021-08-21-11-08_topic30

**Log：**gc_atm/20news_clean_2021-08-21-11-08_topic30

**Method**：并且**单独设置了Generator的学习率为：lr/100**，**改变了temperature为0.5**

**Topic Num:**30

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210821200100183](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210821200100183.png)

**Time：**20news_clean_2021-08-21-22-51_topic20

**Log：**gc_atm/20news_clean_2021-08-21-22-51_topic20

**Method**：并且**单独设置了Generator的学习率为：lr/100**，**改变了temperature为0.5**

**Topic Num:**20

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210822095626787](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210822095626787.png)



### 使用平均加权的label Embedding

**Time：**20news_clean_2021-08-23-12-04_topic20

**Log：**gc_atm/20news_clean_2021-08-23-12-04_topic20

**Method**：并且**单独设置了Generator的学习率为：lr/100**，**改变了temperature为0.5**

**Topic Num:**20

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210823140015605](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210823140015605.png)

---

**Time：**20news_clean_2021-08-27-12-28_topic30

**Log：**gc_atm/20news_clean_2021-08-27-12-28_topic30

**Method**：并且**单独设置了Generator的学习率为：lr/100**，**改变了temperature为0.5**

**Topic Num:**20

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210827194625325](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210827194625325.png)

---

**Time：**20news_clean_2021-08-23-14-54_topic100

**Log：**gc_atm/20news_clean_2021-08-23-14-54_topic100

**Method**：并且**单独设置了Generator的学习率为：lr**，**改变了temperature为0.5**

**Topic Num:**100

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210824092439273](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210824092439273.png)

Batch Size:1024  max最大值0.03013 log：gc_atm/20news_clean_2021-08-24-23-21_topic100/

---

**Time：**20news_clean_2021-08-26-10-09_topic75

**Log：**gc_atm/20news_clean_2021-08-26-10-09_topic75

**Method**：并且**单独设置了Generator的学习率为：lr**，**改变了temperature为0.5**

**Topic Num:**75

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210826210817356](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210826210817356.png)

---

**Time：**20news_clean_2021-08-26-22-19_topic50

**Log：**gc_atm/20news_clean_2021-08-26-22-19_topic50

**Method**：并且**单独设置了Generator的学习率为：lr**，**改变了temperature为0.5**

**Topic Num:**50

**改变超参数如下：**

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |     1e-4      |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

![image-20210827094559319](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210827094559319.png)

---

| batch size | clip | learning rate | beta_1 | beta_2 | n_critic | optimizer | temperature |
| :--------: | :--: | :-----------: | :----: | :----: | :------: | :-------: | :---------: |
|    256     | 0.01 |   **1e-6**    |  0.5   | 0.999  |    5     |   Adam    |   **0.5**   |

效果很差：

![image-20210827121935181](https://gitee.com/yxbLovewy/my-pictures/raw/master/image-20210827121935181.png)

---

