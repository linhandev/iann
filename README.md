# iann

交互式标注软件

## 训练

1. 在需要训练自己的数据集时，首先需要将数据集构造为如下格式，然后放入datasets文件夹中。这里使用的是davis数据集的构造方式。

```
DatasetName
     ├── img
     |    └── filename.jpg
     └── gt
          └── filename.png
```

2. 接下来在ritm_train.py中修改文件的batchsizel什么的，然后在model的区域将不存在的加载模型注释掉，在数据区修改SBDDataset为DavisDataset，换上自己的数据路径，基本直接运行ritm_train.py就可以训练自己的数据集了。【这里后面我觉得可以改一下，将训练的设置做成配置文件的方式或者终端传参那种方式，方便用户训练，现在这样还是不便于用户训练自己的模型。不过先测试过了再看】
3. 感觉这个gif很形象，先搞下来贴上再说。

![248cf21130732e7dfa5fcf515b6f6217](C:\Users\Geoyee\Desktop\248cf21130732e7dfa5fcf515b6f6217.gif)

### *手部数据集进行实验

- 这里有个坑，数据不能有没有标签的纯背景，这样找不到正样点训练就会卡住，并且还不报错。
- 数据集放在AI studio了：https://aistudio.baidu.com/aistudio/datasetdetail/83970，datasets中就留空的（方便github的上传下载）
- 只简单训练了1000个iters，2个epoch都没完，效果很，就离谱- -。

![image-20210425165312752](C:\Users\Geoyee\AppData\Roaming\Typora\typora-user-images\image-20210425165312752.png)

