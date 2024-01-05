import os,re
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtCore import QPointF, Qt,QRect,QPoint
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from numpy.random.mtrand import multivariate_normal


from sklearn.cluster import KMeans

from paddleocr import PaddleOCR
import pandas as pd
from PIL import ImageColor
import colorsys

from tqdm import tqdm

from ColorSelector import _qcolortoHEX,_hextoRGB
from sklearn.metrics import silhouette_score

def color_clustering(img,n=2):
    '''移植
    
    # 哇哇哇啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊哇哇啊啊啊啊啊啊啊啊啊啊啊啊啊啊'''
    # Load image
    # img = cv2.imread(img_path)

    # Convert image to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Reshape image to a 2D array of pixels
    pixels = img.reshape(-1, 3)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n+1, random_state=337845818,).fit(pixels)

    # Get the cluster centers
    colors_rgb = kmeans.cluster_centers_

    # Convert colors to HSV
    colors_hsv = []
    for color in colors_rgb:
        hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
        colors_hsv.append(hsv)

    # Get the labels for each pixel
    labels = kmeans.labels_

    # Count the number of pixels in each cluster
    counts = [0] * len(colors_rgb)
    for label in labels:
        counts[label] += 1

    # Sort the colors by count
    colors_sorted = [colors_hsv[i] for i in reversed(sorted(range(len(counts)), key=lambda k: counts[k]))]

    print(pd.concat([
        pd.DataFrame([cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0] for hsv in colors_sorted]).rename({0:"R",1:"G",2:"B"},axis=1),
        pd.DataFrame(colors_sorted).rename({0:"H",1:"S",2:"V"},axis=1),
        pd.DataFrame([{"percentage":count/sum(counts)} for count in counts]).applymap(lambda s:f"{s:.3%}")
        ],axis=1)[1:].to_markdown())
    #print(colors_sorted)

    # Calculate silhouette score
    #silhouette_avg = silhouette_score(pixels, labels)

    return colors_sorted[1:]                #, silhouette_avg


def ocr(img): #识别图像，返回横坐标
    # 灰度化和二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # OCR识别
    #data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    ocr = PaddleOCR(use_angle_cls=True,use_gpu=True) #, lang="ch"
    result = ocr.ocr(thresh, cls=True) 
    
    return result

def ocr_x(result): #识别图像，返回横坐标
    
    x_list=[]
    cpl_x_list=[]
    for line in result[0]:
        # x = re.findall(r'^[1-9][0-9]{2}', str(line[1][0]))
        #print(pd.DataFrame(line).to_markdown())    
        x = re.findall(r'\b[1-9]\d{2,3}\b', str(line[1][0])) #三位数以及四位数
        #print(x)
        if x:
            x_list.append((int((line[0][0][0] + line[0][1][0]) / 2),int(x[0])))
        # x = re.findall(r'^[1-9][0-9]{2}', str(line[1][0]))
        cpl_x = re.findall(r'\b\d+\b', str(line[1][0])) #啥位数都匹配
        #print(x)
        if cpl_x:
            cpl_x_list.append((int((line[0][0][0] + line[0][1][0]) / 2),cpl_x[0]))

    return x_list, cpl_x_list



def ocr_y(result): #识别图像，返回横坐标
    y_list=[]
    cpl_y_list=[]
    for line in result[0]:
        # x = re.findall(r'^[1-9][0-9]{2}', str(line[1][0]))
        y = re.findall(r'\b[0-9](?:.|,)[0-9]{,3}\b', str(line[1][0])) #三位数以及四位数
        #print(y,int((line[0][0][1] + line[0][1][1]) / 2))
        if y and float(y[0].replace(",","."))<2:
            y_list.append((int((line[0][0][1] + line[0][1][1]) / 2),float(y[0].replace(",","."))))

    return y_list   #,cpl_x_list



class ImgFigure:
    def __init__(self,img):
        if type(img)==type("   ") and os.path.exists(img):
            img = cv2.imread(img) 
        self.img=img
        try:
            self._oldcrop=self.img_crop()  or (1,2,3,4,5,6,7,8)
        except Exception as e:
            print("Find nothing from image")
            print(f"[{e.__class__.__name__}] {e}")
            self._oldcrop=(1,2,3,4,5,6,7,8)
        self._setcap(self._oldcrop)
        #self.find_line(6)

    @classmethod
    def fromQImage(cls, qimage: QImage):
        # 获取 QImage 的宽度和高度
        width = qimage.width()
        height = qimage.height()

        # 将 QImage 转换为 numpy 数组
        buffer = qimage.constBits()
        buffer.setsize(qimage.sizeInBytes())  # 设置缓冲区大小
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

        # 将 numpy 数组转换为 OpenCV 图像
        img = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

        # 创建并返回 ImgFigure 对象
        return cls(img)


    
    def getWidth(self,img=None):
        if img is None: img=self.img.copy()
        height, width, channel = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).shape
        return width
    
    def getHeight(self,img=None):
        if img is None: img=self.img.copy()
        height, width, channel = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).shape
        return height
    
    def getBytesPerLine(self,img=None):
        if img is None: img=self.img.copy()
        height, width, channel = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).shape
        return channel * width
    
    def remapImg(self,  x1,y1,x2,y2,
                        x1_value,y1_value,x2_value,y2_value):
        x1,x2=max(0,min(self.getWidth()-1,x1,x2)),min(self.getWidth()-1,max(0,x1,x2))
        y1,y2=min(self.getHeight()-1,max(0,y1,y2)),max(0,min(self.getHeight()-1,y1,y2))
        x1_value,x2_value=min(x1_value,x2_value),max(x1_value,x2_value)
        y1_value,y2_value=min(y1_value,y2_value),max(y1_value,y2_value)
        
        self._oldcrop=(x1,x2,x1_value,x2_value,
                      y1,y2,y1_value,y2_value)
        self._setcap(self._oldcrop)

    def recapImg(self,x1,x2,y1,y2):
        self._setcap(self._oldcrop)
        x1,x2=max(0,min(self.getWidth()-1,x1,x2)),min(self.getWidth()-1,max(0,x1,x2))
        y1,y2=max(0,min(self.getHeight()-1,y1,y2)),min(self.getHeight()-1,max(0,y1,y2))
        if x1==x2 or y1==y2 :return 
        x1_value=self.value_x/self.num_x*(x1-self.x1)+self.x1_value
        x2_value=self.value_x/self.num_x*(x2-self.x2)+self.x2_value
        y1_value=self.value_y/self.num_y*(y1-self.y1)+self.y1_value
        y2_value=self.value_y/self.num_y*(y2-self.y2)+self.y2_value
        self._setcap((x1,x2,x1_value,x2_value,
                      y1,y2,y1_value,y2_value))
        pass
    def _setcap(self,xy):
        self.x1,self.x2,self.x1_value,self.x2_value,\
        self.y1,self.y2,self.y1_value,self.y2_value=xy

        self.num_y=self.y2-self.y1
        self.value_y=self.y2_value-self.y1_value
        
        self.num_x=self.x2-self.x1
        self.value_x=self.x2_value-self.x1_value
    
    def img_crop(self,img=None):
        if img is None: img=self.img.copy()
        
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # 二值化 反转, 白色为0，黑色为1
        _, bw = cv2.threshold(gray, 225, 1, cv2.THRESH_BINARY_INV) #225,点比较多

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        height, width = edges.shape[:2]
        limit=min(height, width) * 0.4 #识别边界，主要是y
        # 检测直线
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=limit, maxLineGap=10)



        #返回要裁剪的纵坐标y1,y2
        y1list=[0]
        y2list=[height]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1==y2 :
                if y1 < height/2:
                    y1list.append(y1)
                else:
                    y2list.append(y2)
        y1_crop=max(y1list)
        y2_crop=min(y2list)  

        #返回要裁剪的横坐标像素x1 x2，以及对应的波长x_value
        ocred=ocr(img)
        x_list,cpl_x_list=ocr_x(ocred)
        y_list=ocr_y(ocred)

        print("matched")
        print(pd.DataFrame(cpl_x_list).rename({0:"x",1:"OCR value"},axis=1).to_markdown())
        print("\nfiltered")
        print(pd.DataFrame(x_list).rename({0:"x",1:"OCR value"},axis=1).to_markdown())
        print("\nfiltered y")
        print(pd.DataFrame(y_list).rename({0:"y",1:"OCR value"},axis=1).to_markdown())

        try:
            x_array=np.array(x_list)
            #print(x_list)
            x1=min(x_array[:][:,0])
            x2=max(x_array[:][:,0])
            index_1=np.where(x_array[:][:,0]==x1)[0][0]
            index_2=np.where(x_array[:][:,0]==x2)[0][0]
            x1_value=x_array[index_1][1]
            x2_value=x_array[index_2][1]

            y_array=np.array(y_list)
            y1=min(y_array[:][:,0])
            y2=max(y_array[:][:,0])
            index_1=np.where(y_array[:][:,0]==y1)[0][0]
            index_2=np.where(y_array[:][:,0]==y2)[0][0]
            y1_value=y_array[index_1][1]
            y2_value=y_array[index_2][1]
            


            y1,y2=int(y1),int(y2)
            return x1,x2,x1_value,x2_value,y1,y2,y1_value,y2_value
        except IndexError as e:
            print(f"error No data parsed:[{e.__class__.__name__}] {e}")
            return
        except Exception as e:
            print(f"error in image crop:[{e.__class__.__name__}] {e}")
            raise
    
    def find_color_num(self,max_num=10,img=None,cover_gray=True):
        if img is None: img=self.img.copy()
        if cover_gray: img=self._cover_gray(img=img)
        img_cut=img[self.y1+int(0.00*(self.y2-self.y1)):
                        self.y2-int(0.00*(self.y2-self.y1)),
                        self.x1+int(0.00*(self.x2-self.x1)):
                        self.x2-int(0.00*(self.x2-self.x1))]
        
        def is_color_similar(hsv1, hsv2, hue_threshold=5, saturation_threshold=5, value_threshold=5):
            hue_diff = min(abs(hsv1[0] - hsv2[0]), 360 - abs(hsv1[0] - hsv2[0]))
            saturation_diff = abs(hsv1[1] - hsv2[1])
            value_diff = abs(hsv1[2] - hsv2[2])
            h, s, v = hsv1
            if v < 46 or h + s + v < 60:
                #print('black')
                return True
            elif s < 43 and v > 46:
                #print('gray')
                return saturation_diff < saturation_threshold
            else:
                #print('?')
                return hue_diff < hue_threshold

        unique_colors_lengths = []
        for n in range(1,max_num+1):
            colors=color_clustering(img_cut,n)
            
            unique_colors = []
            for i in range(len(colors)):
                is_unique = True
                for j in range(i+1, len(colors)):
                    print(f"{colors[i]} | {colors[j]} : {is_color_similar(colors[i], colors[j])}")
                    if is_color_similar(colors[i], colors[j]):
                        is_unique = False
                        break
                if is_unique:
                    unique_colors.append(colors[i])

            print(f"count:{len(unique_colors)}")
            if len(unique_colors)!=len(colors):
                unique_colors_lengths.append(len(unique_colors))
            #unique_colors_lengths.append(len(unique_colors))
        
        #result = max(set(unique_colors_lengths), key=unique_colors_lengths.count)
        result = min(set(unique_colors_lengths))
        return result
        

    def _cover_gray(self,img=None):
        if img is None: img=self.img.copy()
        # 将图像从BGR颜色空间转换到灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 设置浅灰色的阈值范围
        low_gray = 200
        high_gray = 250

        # 对图像进行阈值处理
        img=cv2.bitwise_not(img)
        mask = cv2.inRange(gray_img, low_gray, high_gray)

        # 去除浅灰色
        img = cv2.bitwise_and(img, img, mask=~mask)
        img=cv2.bitwise_not(img)
        return img
    
    def find_line(self,num_line=3,img=None,cover_gray=True):
        if img is None: img=self.img.copy()
        if cover_gray: img=self._cover_gray(img=img)
                

        n=num_line
        # count=0
        zuobiao=np.zeros((self.num_x,n+1))
        #提取颜色
        img_cut=img[self.y1+int(0.00*(self.y2-self.y1)):
                        self.y2-int(0.00*(self.y2-self.y1)),
                        self.x1+int(0.00*(self.x2-self.x1)):
                        self.x2-int(0.00*(self.x2-self.x1))]
        colors=color_clustering(img_cut,n)
        print(pd.DataFrame(colors).to_markdown())
        # colors=newcolor(img[y1:y2,x1:x2],n)

        #print(f"sil_score:{pd.DataFrame({i:color_clustering(img_cut,i) for i in tqdm(range(2,11))}).to_markdown()}")
        color_range=np.zeros((2*n,3))

        color_img = np.zeros((100, 100, 3), np.uint8)
        '''长得像层析色谱的东西'''
        # 画色图()
        top_colors=colors
        for i, color in enumerate(top_colors):
            color_rgb = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0] #很重要
            #color_rgb=ImageColor.getrgb(f"hsv({color[0]},{color[1]/255:%},{color[2]/255:%})")
            a=100//len(top_colors)
            color_img[:, i*a:(i+1)*a] = color_rgb
                
                #提取颜色对应的曲线
        img1=img.copy()[self.y1+int(0.00*(self.y2-self.y1)):
                        self.y2-int(0.00*(self.y2-self.y1)),
                        self.x1+int(0.00*(self.x2-self.x1)):
                        self.x2-int(0.00*(self.x2-self.x1))]
        blank_image = 255 * np.ones(img1.shape, np.uint8)
        


        def mymean(data):
            '''移植来的不知道什么神秘函数'''
            data=np.array(data)
            if len(data) > 3: #调参的魅力；有些线太细了
                '''看看细线'''
                mean0=np.mean(data)
                fang=np.std(data)
                # data1=data.all(int(mean0-fang)<data<int(mean0+fang))
                data1=data[(mean0-fang<data)&(data<mean0+fang)]
                mean1=np.mean(data1)
                return mean1        
            else:
                return np.mean(data) #空值返回1，对应0

        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

        #count=0
        point_map={}
        for i in range(len(colors)):
            print(colors[i])
            h,s,v=colors[i]
            print("rgb")
            print(ImageColor.getrgb(f"hsv({h*2},{s/255:%},{v/255:%})"))
            hexRGB='#{:02x}{:02x}{:02x}'.format(*ImageColor.getrgb(f"hsv({h*2},{s/255:%},{v/255:%})"))
            point_map.setdefault(hexRGB,{})
            '''if v<46 or h+s+v < 60:
                lower=np.array([0,0,0])
                upper=np.array([180,255,46])
                print('black')
            elif s<43 and v>46:
                lower=np.array([0,0,46])
                upper=np.array([180,46,220])
                print('gray')
            else:
                lower=np.array([max(0,h-10),43,46])
                upper=np.array([min(h+10,180),255,255])
            mask = cv2.inRange(hsv, lower, upper)
            # print(lower, upper)
'''
            mask=self.getColorMask(img=img1,color=hexRGB)
            data=np.where(mask!=0)
            x=data[1]
            y=data[0]
            if len(y)==0:
                continue
            xishu=self.value_x/self.num_x
            res_x=xishu * x + self.x1_value #x变换为标准波长

            zhishu=self.value_y/self.num_y 
            res_y=zhishu * y + self.y1_value #y变化，主要是将底部归零



            unique_nums = np.unique(res_x) 
            zuobiao=[]
            for num in unique_nums:
                # rows = y[res_x == num] # 获取第一列数字等于num的行的第二列数据
                rows = res_y[res_x == num] # 获取第一列数字等于num的行的第二列数据
                # avg=max(rows)
                avg = mymean(rows) # 计算第二列的平均值
                # avg = np.mean(rows) # 计算第二列的平均值
                zuobiao.append((int(num),avg))

            # plt.scatter([d[0] for d in zuobiao], [d[1] for d in zuobiao])
            dianzhen=np.array(zuobiao).T
            # dianzhen=np.array(sorted(zuobiao, key=lambda x: x[0], reverse=True)).T #没用的代码
            value=dianzhen[1]
            sum_abs_diff = 0  # 初始和为0
            for i in range(len(value)-1):
                sum_abs_diff += abs(value[i] - value[i+1])  # 计算相邻两个数之间的绝对值查并累加到和中
            print('disorder:'+str(sum_abs_diff))  #
            if sum_abs_diff <50:
                '''plt.scatter(dianzhen[0],dianzhen[1],c=[np.array(ImageColor.getrgb(f"hsv({h*2},{s/255:%},{v/255:%})"))/255])
                # plt.show()  #唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt唉plt
                count+=1'''
                point_map[hexRGB].update({x:y for x,y in zuobiao})
                # save_to_csv(dianzhen[0], dianzhen[1], img_path,count) #保存为csv
                
            # save_to_csv(res_x, res_y, img_path,i) #保存为csv #没用
        '''fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        img_rgb=cv2.cvtColor(img[self.y1+int(0.05*(self.y2-self.y1)):
                        self.y2-int(0.00*(self.y2-self.y1)),
                        self.x1+int(0.00*(self.x2-self.x1)):
                        self.x2-int(0.00*(self.x2-self.x1))], cv2.COLOR_BGR2RGB) #也很重要
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(color_img)
        ax2.set_title('Top {} Colors'.format(n))
        ax2.axis('off')
        plt.show()
'''

        #print(pd.DataFrame(point_map).to_markdown())
        return pd.DataFrame(point_map)
    
    def getColorMask(self,img=None,color:QColor|tuple[int,int,int]|str="#ffffff",threshold=5):
        if img is None: img=self.img.copy()

        if type(color)==type(QColor()): color=_qcolortoHEX(color)
        if type(color)==type(""): color=_hextoRGB(color)


        color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
        #color=colorsys.rgb_to_hsv(*(np.array(color)/255))
        #color=(np.array(color)*255).astype(np.uint8)

        hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v=tuple(color)
        if v<46 or h+s+v < 60:
                lower=np.array([0,0,0])
                upper=np.array([180,255,46])
                #print('black')
        elif s<43 and v>46:
                lower=np.array([0,0,46])
                upper=np.array([180,46,220])
                #print('gray')
        else:
                lower=np.array([max(0,h-threshold),43,46])
                upper=np.array([min(h+threshold,180),255,255])
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    def getColorMaskHSV(self,img=None,colorhsv=(0,255,255),threshold=5):
        h,s,v=colorhsv
        return self.getColorMask(img=img,color=ImageColor.getrgb(f"hsv({h*2},{s/255:%},{v/255:%})"),threshold=threshold)
    
    #什么时候就做返回颜色遮罩哇啊啊啊啊啊啊啊啊啊啊哇啊啊啊啊啊啊啊啊啊哇啊啊啊啊啊啊啊啊啊啊啊哇啊哇哇啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
    #大卸八块搬迁一下上面已经有的东西
    def getMaskedImg(self,color:QColor|tuple[int,int,int]|str="#ffffff",threshold=5):
        img=self.img.copy()[self.y1:self.y2,
                            self.x1:self.x2]
        mask=self.getColorMask(img,color,threshold)
        img=cv2.bitwise_and(img, img, mask=mask)
        img = np.ones_like(img) * 255 - img
        img1=self.img.copy()
        img1[self.y1:self.y2,self.x1:self.x2]=img
        '''mask = np.all(img == (0, 0, 0), axis=2)
        img[mask] = (255, 255, 255)'''
        return img1


