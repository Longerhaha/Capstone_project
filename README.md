# 机器学习纳米学位
## 毕业项目：猫狗大战
## 2016.5
## 1、软件与库：
    操作系统：win10,64bit
    GPU：GTX1070
    库文件：keras,graphviz,tensorflow-gpu,python3.5常用的一些库
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
### 2.3、图片的尺寸：
  我们可以从训练集可视化图片（下图）看出输入的图片大小不一样，所以我们要使用keras中的[图片生成器](https://keras.io/preprocessing/image/),其能够将猫狗图片reshape成一样大小的图片。
  
    ![train_pic_visualization](https://github.com/Longerhaha/Capstone_project/blob/master/image_file/train_pic_visualization.png)
### 上述的运行结果均可以从该[notebook文件](https://github.com/Longerhaha/Capstone_project/blob/master/Test%20picture%20sequence.ipynb)获得。
## 3、数据预处理
   keras的图片生成器有个从文件夹生成的办法，其函数是flow_from_directory。这个函数把要操作的文件夹的子目录分别当做一个类，根据这个思路我们可以对数据文件进行预处理，将猫、狗分别归于一个文件夹。由于在windows10下获取文件符号链接权限非常麻烦，所以另辟蹊径。在其他操作系统上可以参考[该文](https://github.com/ypwhs/dogs_vs_cats)
  
    本文采用shutil.move的方式对train、test文件夹进行预处理
    #修改到你的数据文件夹目录
    root_src = 'D:\Dogs vs. Cats'
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
## 获取该图像经过预训练模型后的图像特征参数
