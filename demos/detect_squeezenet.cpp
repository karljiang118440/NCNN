#include <stdio.h>
#include <algorithm>
#include <vector>
#include"gesture.id.h"


#include "net.h"

//使用ncnn，传入的参数第一个是你需要预测的数据，第二个参数是各个类别的得分vector，注意传入的是地址，这样才能在这个函数中改变其值
static int detect_squeezenet( float *data, std::vector<float>& cls_scores)
{
    //实例化ncnn：Net，注意include "net.h"，不要在意这时候因为找不到net.h文件而include<net.h>报错，后文会介绍正确的打开方式
    ncnn::Net squeezenet;
    //加载二进制文件，也是照写，后面会介绍对应文件应该放的正确位置
    int a=squeezenet.load_param("demo.param");
    int b=squeezenet.load_param_bin("demo.bin");
    //实例化Mat，前三个参数是维度，第四个参数是传入的data，维度的设置根据你自己的数据进行设置，顺序是w、h、c
    ncnn::Mat in = ncnn::Mat(550,  8, 2, data);

    //实例化Extractor
    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);
    //注意把"data"换成你deploy中的数据层名字
    int d= ex.input("data", in);

    ncnn::Mat out;
    //这里是真正的终点，不多说了，只能仰天膜拜nihui大牛，重点是将prob换成你deploy中最后一层的名字
    int c=ex.extract("prob", out);

    //将out中的值转化为我们的cls_scores，这样就可以返回不同类别的得分了
    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {

        cls_scores[j] = out[j];
    }

    return 0;
}
int main(int argc, char** argv)
{
    //注意，这里的argv是之后从终端输入的参数，我这里是数据源的路径,因为我是从两个文件中生成一个总的数据，所以用了argv[1]和argv[2]，你也可以自己根据需求改变
    const char* imagepath1 = argv[1];
    const char* imagepath2=argv[2];

    FILE *fopeni=NULL;
    FILE *fopenq=NULL;

    fopeni=fopen(imagepath1,"r");
    fopenq=fopen(imagepath2,"r");

    //这是我的数据，i和q相当于图片的两个通道
    float i[4400];
    float q[4400];
    float data[8800];



    int count=4400;
    for (int j = 0; j < count; ++j)
    {
      fscanf(fopeni,"%f",&i[j]);
      fscanf(fopenq,"%f",&q[j]);
    }   
    //这是将iq（相当于图片的两个通道的数据）转化为一个一维向量，需要特别注意的是数据维度的顺序
    for (int j = 0; j < 8800; ++j)
    {
        if (j<4400)
        {
            data[j]=i[j];
        }else{
            data[j]=q[j-4400];
        }

    }



    char a[13]={'A','B','C','F','G','H','I','J','K','L','M','N','O'};

    //注意，这里是调用ncnn的代码
    std::vector<float> cls_scores;//用来存储最终各类别的得分
    //这个函数的实现在上面，快去看
    detect_squeezenet(data, cls_scores);
    for (int i = 0; i < cls_scores.size(); ++i)
    {
         printf("%c : %f\n", a[i],cls_scores[i]);
    }


    return 0;
}


