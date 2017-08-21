# 机器学习纳米学位
## 毕业项目：猫狗大战
## 2017.5-2017.8
## 1、软件与库：
    操作系统：win10,64bit
    GPU：GTX1070
    库文件：keras,graphviz,tensorflow-gpu,python3.5及python常用的一些库
## 2、数据来源
    这个项目来自kaggle的一个比赛[猫狗大战](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)已经过期了，但我们可以提交文件，然后预测loss,最后可以根据现有的公有排名预测自己的名词。在获得数据之后，需要注意的地方如下。
### 2.1、图片序数与window图像文件夹下看到的不一样
    训练集：
    cat.0.jpg
    cat.1.jpg
    cat.10.jpg
    cat.100.jpg
    cat.1000.jpg
    cat.10000.jpg
    cat.10001.jpg
    cat.10002.jpg
    cat.10003.jpg
    cat.10004.jpg
    测试集：
    1.jpg
    10.jpg
    100.jpg
    1000.jpg
    10000.jpg
    10001.jpg
    10002.jpg
    10003.jpg
    10004.jpg
    10005.jpg
    
### 2.2、训练集、测试集大小
    the number of train picture file is 25000
    the number of test picture file is 12500
### 2.3、图片的尺寸
    我们可以从训练集可视化图片（下图）看出输入的图片大小不一样，所以我们要使用keras中的[图片生成器](https://keras.io/preprocessing/image/),其能 够将猫狗图片reshape成一样大小的图片。
    ![train_pic_visualization](https://github.com/Longerhaha/Capstone_project/blob/master/image_file/train_pic_visualization.png)
    上述的运行结果均可以从该[Test picture sequence.ipynb](https://github.com/Longerhaha/Capstone_project/blob/master/Test%20picture%20sequence.ipynb)文件获得。
## 3、数据预处理
   keras的图片生成器有个从文件夹生成的办法，其函数是flow_from_directory。这个函数把要操作的文件夹的子目录分别当做一个类，根据这个思路我们可以对数据文件进行预处理，将猫、狗分别归于一个文件夹。由于在windows10下获取文件符号链接权限非常麻烦，所以另辟蹊径。在其他操作系统上可以参考[该文](https://github.com/ypwhs/dogs_vs_cats)
  
    本文采用shutil.move的方式对train、test文件夹进行预处理
    
    root_src = 'D:\Dogs vs. Cats'#修改到你的数据文件夹目录
    import os
    import shutil
    import time
    os.chdir(root_src)
    #创建训练数据所需的文件夹
    def rmrf_mkdir(dirname,ctl='train'):
        start = time.time()
        if os.path.exists(dirname):
            pass
        else:
            if ctl=='train':
                os.mkdir(dirname)
                os.mkdir('train2/cat')
                os.mkdir('train2/dog')
                #获取当前狗和猫图片的文件名
                train_filenames = os.listdir('train')
                train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
                train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)
                #创建train2文件夹，并将cat类和dog类图片归类到不同的文件夹中
                for filename in train_cat:
                    shutil.move('./train/'+filename, './train2/cat/'+filename)

                for filename in train_dog:
                    shutil.move('./train/'+filename, './train2/dog/'+filename)
            elif ctl=='test':
                test_filenames = os.listdir(dirname.split('/')[0])
                os.mkdir(dirname)
                for filename in test_filenames:
                    shutil.move('test/'+filename,dirname+'/'+filename)
            else:
                pass
        end = time.time()
        print('path deal time is %.5f'%round(end-start,5))

    rmrf_mkdir('train2',ctl='train')  
    rmrf_mkdir('test/test2',ctl='test')
    
    其最终文件夹结构如下：
    ├── test 
        ├── test2 [12500 images]
    ├── test.zip    
    ├── train 
    ├── train.zip
    └── train2
        ├── cat [12500 images]
        └── dog [12500 images]
## 4、模型预训练
    在[Pre_deal of Incep&Xcep&Res.ipynb](https://github.com/Longerhaha/Capstone_project/blob/master/Pre_deal%20of%20Incep%26Xcep%26Res.ipynb)文件，我们通过预处理获得猫狗图像经过Inception、Xception、ResNet50网络后的特征参数，时间大约花了40分钟。在[Pre_deal of VGG16&VGG.ipynb](https://github.com/Longerhaha/Capstone_project/blob/master/Pre_deal%20of%20VGG16%26VGG.ipynb)文件，我们通过预处理获得猫狗图像经过VGG16、VGG19网络后的特征参数。时间大约花了12分钟。
    
    选取这五个模型的最直接原因就是keras里面有这个模型，其次这些模型的深度足够深，参数相对于其他模型比较小，比较占优。预处理函数如下
    
    def write_gap(MODEL, image_size, lambda_func=None):
        width = image_size[0]
        height = image_size[1]
        input_tensor = Input((height, width, 3))
        x = input_tensor
        if lambda_func:
            x = Lambda(lambda_func)(x)

        base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
        model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

        gen = ImageDataGenerator()
        train_generator = gen.flow_from_directory("train2", image_size, shuffle=False, 
                                                  batch_size=16)
        test_generator = gen.flow_from_directory("test", image_size, shuffle=False, 
                                                 batch_size=16, class_mode=None)
        train = model.predict_generator(train_generator, len(train_generator.filenames)/16)
        test = model.predict_generator(test_generator, len(test_generator.filenames)/16)
        print(type(train),train.shape) 
        print(type(test),test.shape) 
        with h5py.File("gap_%s.h5"%(MODEL.__name__)) as h:
            h.create_dataset("train", data=train)
            h.create_dataset("test", data=test)
            h.create_dataset("label", data=train_generator.classes)
    write_gap(ResNet50, (224, 224))
    write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
    write_gap(Xception, (299, 299), xception.preprocess_input)
    
    write_gap(VGG16,(224,224))
    write_gap(VGG19,(224,224))
    
    Xception 、Inception V3 需要将数据限定在 (-1, 1) 的范围内，查keras这两个模型[使用办法](https://github.com/fchollet/keras/tree/master/keras/applications)可知，然后我们利用 GlobalAveragePooling2D 将卷积层平均池化，否则获得的模型参数文件太大且容易过拟合。随后我们定义了train_generator、test_generator，利用 model.predict_generator 函数来导出特征向量，这里要注意，为了遍历所以图片，我们需要设置predict_generator的steps需要设置为len(train(test)_generator.filenames)/16，最后我们选择了 ResNet50, Xception, Inception V3 这三个模型（经测试，联合这三个模型表现较好）。
    
    最后导出的 h5 文件包括三个 numpy 数组：
    train (25000, 2048)
    test (12500, 2048)
    label (25000,)
  该特征参数可从[百度云](http://pan.baidu.com/s/1o7Qvp8m)下载。
## 5、载入特征文件参数
    
    载入特征参数其步骤与读h5文件是一样的，代码如下：
    
    root_src = 'D:\Dogs vs. Cats'
    import os
    import shutil
    import time
    os.chdir(root_src)
    import numpy as np
    import h5py
    import h5py
    import numpy as np
    from sklearn.utils import shuffle
    np.random.seed(2017)

    X_train = []
    X_test = []

    for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['train']))
            X_test.append(np.array(h['test']))
            y_train = np.array(h['label'])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    X_train, y_train = shuffle(X_train, y_train)
    
    ### 打印X_train,y_train的大小
    
    print(X_train.shape,X_test.shape)
    (25000, 6144) (12500, 6144)
    
## 6、模型相关操作
### 6.1 模型构建

    在获取了特征参数后，我们可以直接在其后设置一个全连接网络即可，可能会有部分特征一致或者类似，所以dropout一些特征，另一方面也可以防止过拟合。
    
    from keras.models import *
    from keras.layers import *
    np.random.seed(2017)
    input_tensor = Input(X_train.shape[1:])
    # 可能会有部分特征一致或者类似，所以dropout一些特征，另一方面也可以防止过拟合
    x = Dropout(0.25)(input_tensor)
    x = Dense(1, activation='sigmoid')(x)#直接设置全连接
    model = Model(input_tensor, x)
    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
                  
### 6.2使用graphviz模型可视化
    
    from graphviz import Digraph
    g = Digraph('g',node_attr={'shape': 'record', 'height': '.1'})

    g.node('node0','image')
    g.node('node1','ResNet50|{input:|output:}|{(224, 224, 3)|(2048)} ')
    g.node('node2','InceptionV3|{input:|output:}|{(299, 299, 3)|(2048)}')
    g.node('node3','Xception|{input:|output:}|{(299, 299, 3)|(2048)}')
    g.node('node4','Merge|{input:|output:}|{(3, 2048)|(6144)}')
    g.node('node5','Dropout|{Rate:|input:|output:}|{0.25|(6144)|(6144)}')
    g.node('node6','Output|{input:|output:}|{(6144)|(1)}')

    g.edge('node0','node1')
    g.edge('node0','node2')
    g.edge('node0','node3')
    g.edge('node1','node4')
    g.edge('node2','node4')
    g.edge('node3','node4')
    g.edge('node4','node5')
    g.edge('node5','node6')
    g
    
    ![可视化结果](https://github.com/Longerhaha/Capstone_project/blob/master/image_file/model_visualization.png)
### 6.3 设立停止条件并训练模型
    from keras.callbacks import EarlyStopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2,callbacks=[earlystopping])
 
    Train on 20000 samples, validate on 5000 samples
    Epoch 1/10
    20000/20000 [==============================] - 7s - loss: 0.1056 - acc: 0.9748 - val_loss: 0.0334 - val_acc: 0.9916
    Epoch 2/10
    20000/20000 [==============================] - 4s - loss: 0.0289 - acc: 0.9920 - val_loss: 0.0187 - val_acc: 0.9938
    Epoch 3/10
    20000/20000 [==============================] - 4s - loss: 0.0208 - acc: 0.9940 - val_loss: 0.0154 - val_acc: 0.9950
    Epoch 4/10
    20000/20000 [==============================] - 4s - loss: 0.0178 - acc: 0.9942 - val_loss: 0.0131 - val_acc: 0.9956
    Epoch 5/10
    20000/20000 [==============================] - 4s - loss: 0.0162 - acc: 0.9949 - val_loss: 0.0130 - val_acc: 0.9950
    Epoch 6/10
    20000/20000 [==============================] - 4s - loss: 0.0140 - acc: 0.9960 - val_loss: 0.0120 - val_acc: 0.9958
    Epoch 7/10
    20000/20000 [==============================] - 4s - loss: 0.0140 - acc: 0.9956 - val_loss: 0.0112 - val_acc: 0.9960
    Epoch 8/10
    20000/20000 [==============================] - 4s - loss: 0.0129 - acc: 0.9960 - val_loss: 0.0114 - val_acc: 0.9960
### 6.4 模型调优
#### 6.4.1 模型组合调优
    获得五个模型的预训练特征向量后，由于每个模型都有自己的特点，能获取的特征也不同，所以我们可以尝试将这五个模型的特征向量组合在一起，组合出更好的特征。五个模型的组合方式共有31种，模型组合的预测文件[TestPossibleMerge.ipynb](https://github.com/Longerhaha/Capstone_project/blob/master/TestPossibleMerge.ipynb)：
#### 6.4.2 DropRate调优
    在获取最优的模型组合后，我们继续选择模型中的全连接层drop_rate进行调优。选择了[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]进行调优，预测文件如下：[Test_BestModel_BestDropRate.ipynb](https://github.com/Longerhaha/Capstone_project/blob/master/Test_BestModel_BestDropRate.ipynb)
## 7 根据已调好的参数预测test中的test2图像文件的类别，并生成csv文件,提交到[官网](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
    y_pred = model.predict(X_test, verbose=1)
    y_pred = y_pred.clip(min=0.005, max=0.995)

    import pandas as pd
    from keras.preprocessing.image import *

    df = pd.read_csv("sample_submission.csv")

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory("test", (224, 224), shuffle=False, 
                                             batch_size=16, class_mode=None)
    # 获取文件的序号并减1，然后设置到新预测的csv文件的序号
    for i, fname in enumerate(test_generator.filenames):
        index = int((fname.split('\\')[1]).split('.')[0])
        df.set_value(index-1, 'label', y_pred[i])

    df.to_csv('sample_submission1.csv', index=None)
    df.head(10)
    
    12448/12500 [============================>.] - ETA: 0sFound 12500 images belonging to 1 classes.
    Out[7]:
    id	label
    0	1	0.995
    1	2	0.995
    2	3	0.995
    3	4	0.995
    4	5	0.005
    5	6	0.005
    6	7	0.005
    7	8	0.005
    8	9	0.005
    9	10	0.005
    
    预测这里我们用到了一个小技巧，我们将每个预测值限制到了 [0.005, 0.995] 个区间内，因为kaggle 官方的评估标准是 [LogLoss](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/details/evaluation)，对于预测正确的样本，0.995 和 1 相差无几，但是对于预测错误的样本，0 和 0.005 的差距非常大，是 15 和 2 的差别。参考 [LogLoss 如何处理无穷大问题](https://www.kaggle.com/wiki/LogLoss)，下面的表达式就是二分类问题的 LogLoss 定义。
## 8 项目结果
### 8.1 模型组合测试结果
    将31个模型组合文件提交至Kaggle上预测评分（一天之内只能提交五次），获得的Loss表格如下：
    ![Mode_PossibleMerge_Result](https://github.com/Longerhaha/Capstone_project/blob/master/image_file/Mode_PossibleMerge_Result.png)
    从表格看出，序号为十五的提交文件获得的Loss最低，其对应的模型组合恰好是ResNet50、Xception、InceptionV3的模型组合，这就是为什么我们选择它做我们基准模型的原因。
    虽然0.03867的Loss的评分已经排名十二了，但我不满足于此，因为我确信这三个模型的预训练参数中肯定存在类似的，而我们知道，特征相似太多会导致网络模型过拟合，泛化能力减弱，所以我坚信可以通过调节DropRate来获得更佳的Loss。
### 8.2 DropRate调优测试结果
    将9个DropRate调优文件提交至Kaggle上预测评分（一天之内只能提交五次），获得的Loss表格如下：
    从表格可以看出，当drop_rate = 0.2（对应文件序号1）时，Loss最低。对于这个结果符合我预期。此时排名为第七！冲入前十，心满意足！
## 9 项目总结
    首先要非常感谢这位[同学](https://github.com/ypwhs/dogs_vs_cats)提供的思路。
### 9.1 项目思考
    从这个项目的感悟是：
#### (1)数据预处理一定要做好，这决定着你后续的工作是否能够开展顺利。前期可以做一些图像可视化，去观察图像的特点，再决定着下一步怎么走。
#### (2)可以利用不同模型的特征参数去组合出一个更好的模型，这个思维是以前没有见到过的，也是最有意思的地方。由于不同模型的组合，肯定会有部分特征相似，所以在全连接层要选取一个较好的DorpRate，这可以通过网格搜索或者列表循环实现。
### 9.2 改进方向
    这个项目的改进方向主要有以下几点：
#### (1)可以选择更好的网咯模型导出预训练特征向量，比如InceptionV4、GoogleNet等。
#### (2)可以再获取其他的猫狗图像进行fine-turn。
#### (3)可以进行数据增强、异常数据处理与相似预特征向量处理。

