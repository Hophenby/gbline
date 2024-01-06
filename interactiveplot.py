
import os
import re
import sys
from PyQt6.QtGui import QCloseEvent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from rectlabel import RectLabel


ax,ay,aw,ah=50, 50, 1640, 800

#img_path=r'E:\qqfiles\1094009969\FileRecv\pdfs_2023-11-15_13-00-40\sample\2(4)[6].png'
class InteractivePlot(QMainWindow):
    """
    A QMainWindow subclass for displaying an interactive plot.

    Attributes:
        imgdir (str): The directory path of the image.
        imgname (str): The name of the image file.

    Methods:
        __init__(): Initializes the InteractivePlot instance.
        select_path(): Opens a file dialog to select an image file.
        new_widget(img_path): Creates a new RectLabel widget with the specified image path.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Interactive Plot')
        self.setGeometry(ax,ay,aw,ah)

        '''self.label=QLabel(self)

        self.imgdir=""
        self.imgname=""

        self.select_path()
'''
        try:
            # 创建一个QLabel对象，并将OpenCV图像显示在其中  ,os.path.join(self.imgdir,self.imgname)
            self.label = RectLabel(self)
            #self.label.newfigConnect(self.new_widget)

            layout = QVBoxLayout()
            layout.addWidget(self.label)
            
            widget = QWidget()
            widget.setLayout(layout)
            self.setCentralWidget(widget)

            # 设置鼠标跟踪，使得鼠标移动事件能够触发
            self.label.setMouseTracking(True)
        except:
            self.close()


    def closeEvent(self, a0: QCloseEvent | None) -> None:
        sys.stdout.close()
        # 获取 QTextEdit 中的文本内容
        text = self.label.logbox.toPlainText()

        # 写入日志文件
        with open('app.log', 'a') as f:
            f.write(text)

        # 正常关闭窗口
        a0.accept()
        return super().closeEvent(a0)
'''
    def select_path(self):
        """
        Opens a file dialog to select an image file.

        This method sets the `imgdir` and `imgname` attributes based on the selected file.
        It also calls the `new_widget` method to update the displayed image.
        """
        dialog=QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            
        directory,filetype = dialog.getOpenFileName(self, 'Select Directory',self.imgdir,"img(*)")
        if directory:
            self.imgdir=os.path.join(*(match.group(1) for match in re.finditer(r"(.*)[/\\]", directory)))
            self.imgname=re.search(r"([^/\\]*)$",directory,re.I).group(1)
            self.new_widget(directory)
    
    def new_widget(self,img_path):
        """
        Creates a new RectLabel widget with the specified image path.

        Args:
            img_path (str): The path of the image file.

        This method replaces the current label widget with a new one that displays the specified image.
        """
        widget = self.label  # 从列表中删除最后一个小部件
        widget.setParent(None)  # 将小部件从父级窗口中移除
        widget.deleteLater()  # 删除小部件对象
        newwidget = RectLabel(self,img_path)
        self.setCentralWidget(newwidget)  # 将新的小部件添加到当前布局管理器中
        self.label=newwidget
        self.label.newfigConnect(self.new_widget)
        self.label.setMouseTracking(True)
        self.update()

'''