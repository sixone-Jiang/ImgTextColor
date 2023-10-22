# ImgTextColor: 识别文字图像中 "文字" 的颜色



![image-20231022145246345](https://raw.githubusercontent.com/sixone-Jiang/Picgo/main/image-20231022145246345.png)



## 简介

ImgTextColor: 识别文字图像中 "文字" 的颜色；

要是有帮助到您请不要吝啬您的Star；

如有问题欢迎提交Issues.



## 环境配置

```bash
--python env default is 3.9; window and linux is tested!
--pip install 
			pillow
			numpy
			opencv-python
--server add module:
  pip install fastapi
  			  starlette
  			  uvicorn
  			  python-multipart
			
```



## 快速使用

测试（无需下载server module）：

```bash
python rec_image_color.py
```

测试样例图(默认)--test_image/test.jpg

输出结构：

* shell 中upper_front_asume_color 后接 hex格式颜色
* upload_image中展示了各阶段性输出



**颜色转换工具** ：https://sunpma.com/other/rgb/



启用服务端程序：

```bash
python main.py

访问：
--fastapi: http://localhost:8080/docs
--上传图像/下载推理色块： http://localhost:8080/rec_image_color_upload_show_demo
--上传list[base64_str]/获取list[str(hex_color)]:	http://localhost:8080/rec_image_color_base64_image_list
```



## 实现方案

1. 增强图像（并非灰度化或二值化）

2. Kmeans两分类增强后图像，将图像转换为仅包含两种颜色的图
3. 利用四角和四边线，投票选出背景
4. 分离出前景代表的像素位置列表，对原始图的这些位置Kmeans分一类，得到前景颜色