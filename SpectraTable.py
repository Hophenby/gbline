
import os
import typing
import numpy as np
from figuregrabbing import GrabitLine,GrabitStack, _hextoRGB
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage,QPixmap,QPen,QPainter,QColor,QMouseEvent,QPaintEvent
from PyQt6.QtWidgets import QTableWidget,QTableWidgetItem, QLabel, QFileDialog

from ColorSelector import ColorSelector

class SpectraTable(QTableWidget):
    def __init__(self,parent):
        """
        Initializes the SpectraTable object.

        Args:
            parent: The parent object.

        Returns:
            None
        """
        super().__init__(parent)
        if hasattr(parent,"q_image"): self.q_image=parent.q_image
        self.data=None
        self.commandstack=None
        self.graphPos_=None
        self.previewFig=plt.figure()
        self.canvasLabel=CanvasLabel(parent,self.previewFig)
        self.canvasLabel.setGeometry(900,50,877,666)
        self.colorSelector=None
        self.setGrabbingColor=None
        self.csvdir=""

        self.itemChanged.connect(self.setData)
        
    def connect_colorselector(self,colorSelector:ColorSelector):
        self.colorSelector=colorSelector    
    def connect_grabbingColor(self,grabbingColorSetter):
        self.setGrabbingColor=grabbingColorSetter

    def mousePressEvent(self, e: QMouseEvent | None) -> None:
        index = self.indexAt(e.pos())
        r = index.row()
        c = index.column()
        if e.button() == Qt.MouseButton.RightButton:
            # 获取鼠标右键点击的单元格坐标
            
            try:
                color=self.data.columns[r-1]
                print(color)
                self.setGrabbingColor(color)

            except Exception as e:
                print("failed to select color from table")
                print(f"[{e.__class__.__name__}] {str(e)}")
                return
        if e.button() == Qt.MouseButton.MiddleButton:
            # 获取鼠标右键点击的单元格坐标
            
            try:
                color=self.data.columns[r-1]
                dialog = QFileDialog()
                dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
                dialog.setNameFilter("CSV(*.csv)")

                if os.path.exists(self.csvdir):
                    dialog.setDirectory(self.csvdir)

                dialog.setDefaultSuffix(".csv")
                directory, filetype = dialog.getSaveFileName(self, 'Select Directory', self.csvdir, "CSV(*.csv)")
                if directory:
                    self.rawsavecsv(self.data[[color]],directory)
                    print("data saved")
            except Exception as e:
                print(f"{e.__class__.__name__} : {str(e)}")
                return
        return super().mousePressEvent(e)

    def setData(self, item:QTableWidgetItem) :
        """
        Sets the data for the specified item.

        Args:
            item: The QTableWidgetItem object.

        Returns:
            None
        """
        r,c=item.row(),item.column()
        if c!=0:return
        content=item.text()
        try:
            color=self.data.columns[r-1]
            print(f"content: {content} | {color}")
            self.commandstack.setInfoFromColor(color=color,info=content)
            print(self.commandstack.findInfoFromColor(color))

        except Exception as e:
            print(f"{e.__class__.__name__} : {str(e)}")
            return
        

    def plot_dataframe(self, dataframe:pd.DataFrame,xmin=0,xmax=800,ymin=0,ymax=2,amap:dict={}):
        """
        Plots the dataframe on the figure.

        Args:
            dataframe: The pandas DataFrame object.
            xmin: The minimum x-axis value (default: 0).
            xmax: The maximum x-axis value (default: 800).
            ymin: The minimum y-axis value (default: 0).
            ymax: The maximum y-axis value (default: 2).
            amap: The dictionary containing alpha values for different colors (default: {}).

        Returns:
            None
        """
        self.previewFig.clear()
        ax = self.previewFig.add_subplot(111)
        dfsorted=dataframe.sort_index()         #

        
        defaultAlpha=amap.get("default",255)
        cmap=ListedColormap([(*(np.array(_hextoRGB(hex_))/255),amap.get(hex_,defaultAlpha)/255) for hex_ in dfsorted.columns])
        dfsorted.plot(ax=ax,xticks=np.linspace(xmin,xmax,11),yticks=np.linspace(ymin,ymax,5),legend=False,colormap=cmap)

    def updateData(self,commandStack:GrabitStack=None,graghPos_=None,alphaMap:dict={}):
        """
        Updates the data in the table.

        Args:
            commandStack: The GrabitStack object.
            graghPos_: The graph position function.
            alphaMap: The dictionary containing alpha values for different colors (default: {}).

        Returns:
            None
        """
        self.commandstack=commandStack
        self.graphPos_=graghPos_
        if (self.commandstack is None) or (self.graphPos_ is None) or commandStack.isEmpty():return
        point_map={}
        self.colormap=[]
        for element in (commandStack):
            if element.isEmpty():continue
            color=element.color
            if self.colorSelector and not (color in self.colorSelector):continue
            point_map.setdefault(color,{})
            for point in element:
                (x,y)=graghPos_(point.x(),point.y())
                x=int(x)
                if element.toolmode==0:
                    point_map[color].update({x:y})#笔
                if element.toolmode==1:
                    point_map[color].update({x:np.nan})#橡皮
        try:
            notnan=lambda x,y:not(x is None or y is None or np.isnan(y))
            x_max=max((max({x:y for x,y in v.items() if notnan(x,y)}.keys())for v in point_map.values()))
            x_min=min((min({x:y for x,y in v.items() if notnan(x,y)}.keys())for v in point_map.values()))
            y_max=max((max({x:y for x,y in v.items() if notnan(x,y)}.values())for v in point_map.values()))
            y_min=min((min({x:y for x,y in v.items() if notnan(x,y)}.values())for v in point_map.values()))
            #print(x_min, x_max )
        except Exception as e:
            print(f"{e.__class__.__name__} : {str(e)}")
            return
        self.data=pd.DataFrame(point_map).dropna(how='all')
        #print(self.data.index.min(), self.data.index.max() )

        # 定义填充的步幅
        step_size = 1

        # 获取新的索引
        new_index = np.arange(self.data.index.min(), self.data.index.max() + step_size, step_size)

        # 对DataFrame进行重新索引，并使用线性插值填充缺失值
        self.data = self.data.reindex(new_index).interpolate(method='linear')

        point_map=self.data.to_dict()

        #print(self.data.index.min(), self.data.index.max() )
        #self.setRowCount(x_max-x_min+2)
        #self.setColumnCount(len(point_map)+1)
        self.setColumnCount(x_max-x_min+2)
        self.setRowCount(len(point_map)+1)
        for i,x in enumerate(np.linspace(x_min,x_max,x_max-x_min+1)):
            #self.setItem(i+1,0,QTableWidgetItem(str(x)))
            self.setItem(0,i+1,QTableWidgetItem(str(x)))
        for i,(k,v) in enumerate(point_map.items()):
            #self.setItem(0,i+1,QTableWidgetItem(str(k)))
            kinfo=k
            if self.commandstack.findInfoFromColor(k):kinfo=self.commandstack.findInfoFromColor(k)
            item=QTableWidgetItem(str(kinfo))
            item.setBackground(QColor(*_hextoRGB(k),60))
            self.setItem(i+1,0,item)
            for x,y in v.items():
                #self.setItem(x-x_min+1,i+1,QTableWidgetItem(f"{y:.3f}"))
                y=y or np.nan
                self.setItem(i+1,x-x_min+1,QTableWidgetItem(f"{y:.3f}"))
                self.item(i+1,x-x_min+1).setBackground(QColor(*_hextoRGB(k),60))
            
        if not(self.data is None):
            self.plot_dataframe(self.data,x_min,x_max,y_min,y_max,alphaMap)
        #self.setItem()
    
    '''
    def update(self) -> None:
        self.canvasLabel.update()
        return super().update()'''
    def save_to_csv(self,filepath:str,filename:str,color:str|None=None):
        """
        Saves the data to a CSV file.

        Args:
            filepath: The path to the directory where the file will be saved.
            filename: The name of the file.
            color: The color to be included in the CSV file (default: None).

        Returns:
            bool: True if the file was saved successfully, False otherwise.
        """
        try:
            data=self.data
            if color:
                data=self.data[[color]]
            print({c:self.commandstack.findInfoFromColor(c) or c for c in data.columns})
            if os.path.exists(filepath):
                self.rawsavecsv(data,os.path.join(filepath,filename))
                return True
        except Exception as e:
            print(f"[{e.__class__.__name__}] {str(e)}")
            return
        
    def rawsavecsv(self, data: pd.DataFrame, path):
        """
        Save the given DataFrame as a CSV file.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.
            path (str): The path where the CSV file will be saved.

        Returns:
            None
        """
        data.rename(columns={c: self.commandstack.findInfoFromColor(c) or c for c in data.columns})\
            .to_csv(path, index=True)
        




# 创建一个继承自 QLabel 的部件类
class CanvasLabel(QLabel):
    def __init__(self, parent,figure):
        super().__init__(parent)

        # 将 Figure 嵌入到 FigureCanvasQTAgg 中
        self.canvas = FigureCanvas(figure)

        '''
        # 渲染 FigureCanvas 为 QPixmap
        self.pixmaps = self.render_figure_canvas()

        # 在 QLabel 中显示 QPixmap
        self.setPixmap(self.pixmaps)'''

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter=QPainter(self)
        #painter.drawRect(self.rect())
        try: 
            painter.drawPixmap(0,0,self.render_figure_canvas())
        except Exception as e:
            print(f"{e.__class__.__name__} : {str(e)}")
            

        #print(self.rect().topLeft(),self.rect().bottomRight())
        return super().paintEvent(a0)
    
    def render_figure_canvas(self):
        width, height = self.canvas.get_width_height()
        self.canvas.renderer=self.canvas.get_renderer()
        self.canvas.draw()
        buffer = self.canvas.tostring_rgb()
        #print(buffer)
        image = QImage(buffer, width, height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap
    
