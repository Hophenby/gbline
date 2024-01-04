
import os
import re
import sys
import time
import typing
from PyQt6 import QtGui
from PyQt6 import QtCore
import cv2

from PyQt6.QtWidgets import QBoxLayout, QButtonGroup, QFileDialog, QHBoxLayout, QInputDialog, QMainWindow, QSizePolicy, QSpacerItem, QTextEdit, QVBoxLayout, QApplication,  QWidget, QLabel,QPushButton,QTableWidget,QTableWidgetItem
from PyQt6.QtGui import QCursor, QFont, QGuiApplication, QIcon, QImage, QKeyEvent,QPixmap,QPen,QPainter,QColor,QMouseEvent,QPaintEvent,QClipboard
from PyQt6.QtCore import QEvent, QSize, Qt,QRect,QPoint, pyqtSignal
import numpy as np

from ImgFigure import ImgFigure
from figuregrabbing import GrabitEraserLine, GrabitLine,GrabitStack
from ColorSelector import ColorSelector, _hextoRGB
from SpectraTable import SpectraTable



KEY_HOLDING=1
KEY_RELEASED=0

STATE_DRAG=0
STATE_GRAB=1
STATE_ERASE=2

def make_filename_legal(filename):
    # 替换非法字符
    filename = filename.replace('/', '-')
    filename = filename.replace('\\', '-')
    filename = filename.replace(':', '-')
    filename = filename.replace('*', '-')
    filename = filename.replace('?', '-')
    filename = filename.replace('"', '-')
    filename = filename.replace('<', '-')
    filename = filename.replace('>', '-')
    filename = filename.replace('|', '-')
    
    # 移除多余的点号
    filename = re.sub(r'\.+$', '', filename)
    
    # 移除非法字符后可能导致的空格问题
    filename = filename.strip()
    
    return filename

class EmittingStream(QtCore.QObject):
    """
    A custom stream class that emits a signal whenever text is written.

    Attributes:
        textWritten (pyqtSignal): Signal emitted when text is written.

    Methods:
        write: Writes text and emits the textWritten signal.
    """
    textWritten = pyqtSignal(str)

    def write(self, text):
        """
        Writes the given text and emits the textWritten signal.

        Args:
            text (str): The text to be written.
        """
        self.textWritten.emit(str(text))


class RectLabel(QLabel):
    
    
    def __init__(self,parent,img_path:str=""):
        super().__init__(parent)

        #self.setGeometry(10,10,width+200, height+100)
        self.logbox=QTextEdit(self)
        self.logbox.setFont(QFont("Courier"))
        
        sys.stdout = EmittingStream()
        sys.stdout.textWritten.connect(self.normal_output_written)

        now=str(time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime()))
        self.csvdir=os.path.join(*(match.group(1) for match in re.finditer(r"(.*)[/\\]", img_path)))
        self.csvfilename=now+".csv"

        self.imgFigure=ImgFigure(img_path)
        # 创建一个OpenCV图像
        image = self.imgFigure.img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = channel * width
        self.q_image = QImage(image.data, width, height, bytes_per_line,QImage.Format.Format_RGB888)
        
        #self.xshift,self.yshift=0,0
        self.xshift,self.yshift=150,77
        self.hscale,self.vscale=0.7,0.7
        self.pixmapRect=QRect(self.xshift,self.yshift, 0+int(self.q_image.width()*self.hscale), 0+int(self.q_image.height()*self.vscale))
        self.pixmapRectTemp=QRect(self.pixmapRect)

        self.flag_Dragging = False
        self.dragRect = QRect()
        self.setPixmap(QPixmap.fromImage(self.q_image))

        self.drag_state=0
        self.flag_Dragging = False  # 是否正在拖拽
        self.drag_start_pos = None  # 记录拖拽起始位置
        self.drag_release_pos = None  # 记录拖拽起始位置
        self.mousePos = None
        self.keyStates={}
        self.mouseStates={}

        self.flag_DraggingRight = False  # 是否正在拖拽
        self.dragRight_start_pos = None  # 记录拖拽起始位置

        self.dataWidget=SpectraTable(parent=self)
        self.dataWidget.setGeometry(120,650,1000,200)
        self.saved=True

        parent_x,parent_y=self.mapToParent(self.rect().topLeft()).x(),self.mapToParent(self.rect().topLeft()).y()
        

        self.graphPos_=lambda xcl,ycl:(
            ((xcl-parent_x-self.xshift)/self.hscale-self.imgFigure.x1)/self.imgFigure.num_x*self.imgFigure.value_x+self.imgFigure.x1_value,
            ((ycl-parent_y-self.yshift)/self.vscale-self.imgFigure.y1)/self.imgFigure.num_y*self.imgFigure.value_y+self.imgFigure.y1_value)


        self.parentPos_=lambda xcl,ycl:(
            int(((xcl-self.imgFigure.x1_value)/self.imgFigure.value_x*self.imgFigure.num_x+self.imgFigure.x1)*self.hscale+parent_x+self.xshift),
            int(((ycl-self.imgFigure.y1_value)/self.imgFigure.value_y*self.imgFigure.num_y+self.imgFigure.y1)*self.vscale+parent_y+self.yshift))
        

        self.flag_Grabbing=False
        self.flag_GrabbingRight = False
        self.grabbingColor="#000000"
        self.commandStack=GrabitStack()
        self.lastdrawing=GrabitLine(graghPos_=self.graphPos_)

        
        self.colorSelector=ColorSelector(self,column_mode=True)
        self.colorSelector.setHighlight(self.grabbingColor)
        self.colorSelector.setGeometry(10,460,100,300)
        self.dataWidget.connect_colorselector(self.colorSelector)

        #啊哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊哇啊
        

        '''(
            ((xcl-parent_x-xshift)/hscale-self.imgFigure.x1)/self.imgFigure.num_x*self.imgFigure.value_x+,
            ((ycl-parent_y-yshift)/vscale-self.imgFigure.y1)/self.imgFigure.num_y*self.imgFigure.value_y+self.imgFigure.y1_value)'''
    #TODO 按钮组按钮组按钮组按钮组按钮组按钮组按钮组按钮组按钮组按钮组按钮组按钮组按钮组按钮组    
        self.buttonGroup=QButtonGroup(self)
        for i in range(3):
            button=QPushButton(self)
            button.setGeometry(10,110+70*i,100,50)
            '''button.clicked.connect(lambda x:self.switch_dragEvent(i))#绑的怎么是同一个函数改改改改改改
            print(self.switch_dragEvent(i))
            print(self.drag_state)'''
            modetext={STATE_DRAG:"drag",STATE_GRAB:"grab",STATE_ERASE:"erase"}
            button.setText(f"switch to\n{modetext[i]} mode")
            self.buttonGroup.addButton(button,i)

        self.buttonGroup.button(1).clicked.connect(lambda _:self.switch_dragEvent(1))
        self.buttonGroup.button(2).clicked.connect(lambda _:self.switch_dragEvent(2))
        self.buttonGroup.button(0).clicked.connect(lambda _:self.switch_dragEvent(0))

        self.buttonAnalyse=QPushButton("analyse",self)
        self.buttonAnalyse.setGeometry(10,320,100,50)
        self.buttonAnalyse.clicked.connect(self.analyseEvent)

        self.buttonCsvDir=QPushButton("save csv",self)
        self.buttonCsvDir.setToolTip(f"{self.csvdir}")
        self.buttonCsvDir.setGeometry(10,390,100,50)
        self.buttonCsvDir.mousePressEvent=self.csvdirEvent

        self.buttonAddColor=QPushButton("add color",self)
        self.buttonAddColor.setGeometry(10,390,100,50)
        self.buttonAddColor.clicked.connect(self.colorSelector.appendNew)

        if self.parent():
            self.buttonNew=QPushButton("New figure",self)
            self.buttonNew.setGeometry(10,50,100,50)
            self.buttonNew.clicked.connect(parent.select_path)

        self.installEventFilter(self)
        
        #布局整改 下整改令了
        buttonlayout=QVBoxLayout()
        buttonlayout.setAlignment(Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        buttonlayout.addWidget(self.buttonNew)
        buttonlayout.addWidget(self.buttonAnalyse)
        buttonlayout.addWidget(self.buttonCsvDir)
        for button in self.buttonGroup.buttons():buttonlayout.addWidget(button)
        self.colorSelector.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Expanding)
        buttonlayout.addWidget(self.colorSelector)
        buttonlayout.addWidget(self.buttonAddColor)
        #spacer=QSpacerItem(100,50,QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Fixed)
        #buttonlayout.addItem(spacer)

        tablelayout=QHBoxLayout()
        tablelayout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.dataWidget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Preferred)
        tablelayout.addWidget(self.dataWidget)

        canvaslayout=QHBoxLayout()
        canvaslayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        canvas=self.dataWidget.canvasLabel
        canvas.setFixedSize(626,516)
        canvas.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        canvaslayout.addLayout(buttonlayout)
        canloglayout=QVBoxLayout()
        canloglayout.addWidget(canvas,alignment=Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignRight)
        self.logbox.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Expanding)
        canloglayout.addWidget(self.logbox)
        canvaslayout.addLayout(canloglayout)
        #canvaslayout.addWidget(canvas)
        
        mainlayout=QVBoxLayout()
        #mainlayout.addLayout(buttonlayout)
        mainlayout.addLayout(canvaslayout)
        mainlayout.addLayout(tablelayout)

        self.setLayout(mainlayout)

    def normal_output_written(self, text):
        """
        Writes the given text to the logbox widget, scrolling to the end if necessary.

        Parameters:
        text (str): The text to be written to the logbox.

        Returns:
        None
        """
        cursor = self.logbox.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.logbox.setTextCursor(cursor)
        self.logbox.ensureCursorVisible()

    def csvdirEvent(self, event: QMouseEvent):
        """
        Handle the event when the user interacts with the CSV directory button.

        Args:
            event (QMouseEvent): The mouse event triggered by the user.

        Returns:
            None
        """
        ctrlcolor = ""
        if self.keyStates.get(Qt.Key.Key_Control) == KEY_HOLDING:
            ctrlcolor = self.grabbingColor

        if event.button() == Qt.MouseButton.RightButton:
            dialog = QFileDialog(directory=self.csvdir)
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            dialog.setNameFilter("CSV(*.csv)")

            '''if os.path.exists(self.csvdir):
                dialog.setDirectory(self.csvdir)'''

            dialog.setDefaultSuffix(".csv")

            csvfilename = self.csvfilename
            if ctrlcolor:
                csvfilename = f"{self.csvfilename}-{make_filename_legal(self.commandStack.findInfoFromColor(ctrlcolor) or ctrlcolor)}"

            directory, filetype = dialog.getSaveFileName(self, 'Select Directory', os.path.join(self.csvdir, csvfilename), "CSV(*.csv)")
            self.buttonCsvDir.setToolTip(f"{self.csvdir}")
            if directory:
                self.csvdir = os.path.join(*(match.group(1) for match in re.finditer(r"(.*)[/\\]", directory)))
                self.csvfilename = re.search(r"([^/\\]*\.csv)$", directory, re.I).group(1)
                print(self.csvdir, self.csvfilename)

        if self.dataWidget.save_to_csv(self.csvdir, self.csvfilename, ctrlcolor):
            self.saved = True
            print("data saved")


    def analyse(self, n: int):
        """
        Analyzes the image figure and performs line grabbing based on the specified number of lines.

        Parameters:
        - n (int): The number of lines to analyze.

        Returns:
        None
        """
        analyzed = self.imgFigure.find_line(n)
        firstcolor = None
        for col in analyzed.columns:
            firstcolor = firstcolor or col
            self.commandStack.push(
                GrabitLine(line={self.parentPos_(xcl, ycl)[0]: self.parentPos_(xcl, ycl)[1] for xcl, ycl in analyzed[col].sort_index().to_dict().items() if not np.isnan([xcl, ycl]).any()},
                           color=col,
                           graghPos_=self.graphPos_)
            )
            self.colorSelector.append(col)
            self.dataWidget.updateData(self.commandStack, self.graphPos_)

        self.grabbingColor = firstcolor or self.grabbingColor
        self.colorSelector.setHighlight(self.grabbingColor)

    def analyseEvent(self):
        try:
              text,ok=QInputDialog(self).getText(self,"","count of the lines:",text=f"{self.imgFigure.find_color_num(max_num=6)}")
              if ok:
                   self.analyse(int(text))
                   self.saved=False
        except Exception as e:
             print(f"[{e.__class__.__name__}] {e}")

    def eventFilter(self, obj, event:QEvent):
        if event.type() == QEvent.Type.InputMethod :        #and event.reason() == Qt.InputMethodQuery.ImQueryInput
            # 忽略输入法查询事件
            return True
        
        # 其他事件交给父类处理
        return super().eventFilter(obj, event)


    def getAlphaMap(self):
         
        alphaMap={}
        if self.keyStates.get(Qt.Key.Key_Control):alphaMap={"default":10,self.grabbingColor:255}
        return alphaMap

    def switch_dragEvent(self,state:int):
        self.drag_state=state
        print(f"state:{self.drag_state}")
        self.update()

        '''
        modetext={STATE_DRAG:"drag",STATE_GRAB:"grab",STATE_ERASE:"erase"}
        button.setText(f"switch to\n{modetext[self.drag_state]} mode")'''

    def dragPaintEvent(self,painter:QPainter):
        pen = QPen(QColor(0, 0, 0))
        if self.flag_Dragging:
                pen.setColor(QColor(0, 0, 255))
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                self.drag_release_pos=self.mousePos

            
        if self.flag_DraggingRight:
                pen.setColor(QColor(192, 192, 192))
                pen.setWidth(8)
                pen.setStyle(Qt.PenStyle.DotLine)
                painter.setPen(pen)
                new_pos=self.mousePos-self.dragRight_start_pos
                painter.drawRect(QRect(self.pixmapRect.topLeft()+new_pos,self.pixmapRect.bottomRight()+new_pos))
        else:
                painter.drawRect(self.dragRect)
        '''for x,vx in x_list:
                painter.drawPoint(int(((x-x1_value)*num_x/value_x+xshift)*hscale+x1+9),y2)
                painter.drawText(int(((x-x1_value)*num_x/value_x+xshift)*hscale+x1+9),y2,str(vx))
    '''
        painter.setPen(pen)
        if not (self.drag_release_pos is None):
                xcl,ycl=self.drag_release_pos.x(),self.drag_release_pos.y()
                painter.drawText(self.drag_release_pos,
                                f"({self.graphPos_(xcl,ycl)[0]:.2f}, {self.graphPos_(xcl,ycl)[1]:.2f})")

        if not (self.drag_start_pos is None):
                xcl,ycl=self.drag_start_pos.x(),self.drag_start_pos.y()
                painter.drawText(xcl,ycl-10,
                                f"({self.graphPos_(xcl,ycl)[0]:.2f}, {self.graphPos_(xcl,ycl)[1]:.2f})")
                
        #pen = QPen(Qt.GlobalColor.black,2)
        #painter.setPen(pen)
        #if self.mousePos:
        #    painter.drawRect(self.mousePos.x(),self.mousePos.y(),5,5)
                
    def grabPaintEvent(self,painter:QPainter):

        if self.keyStates.get(Qt.Key.Key_Shift)==KEY_HOLDING:
             painter.drawPixmap(
                  self.pixmapRect,
                  QPixmap.fromImage(QImage(
                       self.imgFigure.getMaskedImg(self.grabbingColor).data, 
                       self.imgFigure.getWidth(), 
                       self.imgFigure.getHeight(), 
                       self.imgFigure.getBytesPerLine(),
                       QImage.Format.Format_RGB888)))
        else:painter.drawPixmap(self.pixmapRect,self.pixmap())
        if self.flag_Grabbing: 
            
            pen = QPen(QColor(*_hextoRGB(self.grabbingColor)),2)
            pen.setStyle(Qt.PenStyle.DotLine)
            if self.flag_GrabbingRight:
                pen = QPen(QColor(222, 222, 222),2)
                pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            self.lastdrawing.display(painter)
        self.commandStack.drawCasting(painter,self.getAlphaMap())

        #pen = QPen(QColor(*_hextoRGB(self.grabbingColor)),2)
        #painter.setPen(pen)
        #painter.drawEllipse(self.mousePos or QPoint(),3,3)
            #哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊

    def paintEvent(self, event):
        
        painter = QPainter(self)
        painter.drawPixmap(self.pixmapRect,self.pixmap())
            
        painter.setPen(QPen(QColor(100,255,100,100),3))
        
        painter.drawRect(self.buttonGroup.button(self.drag_state).geometry())
        

        self.grabPaintEvent(painter)
        if self.drag_state==STATE_DRAG:
            self.dragPaintEvent(painter)
        #return super().paintEvent(event)
        
        try:
            self.show_custom_cursor(event)
        except Exception as e:
            print(f"{e.__class__.__name__} : {str(e)}")

    def show_custom_cursor(self, event):
        # 获取原光标图标对象
        original_cursor = QGuiApplication.overrideCursor()

        # 创建一个新的 QPixmap 对象，并使用 QPainter 绘制一个矩形图标
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        if self.drag_state==STATE_DRAG:
            painter.setPen(QPen(QColor(Qt.GlobalColor.black),2))
            painter.drawLine(0, 16, 31, 16)
            painter.drawLine(16, 0, 16, 31)
        if self.drag_state==STATE_GRAB:
            painter.setPen(QPen(QColor(*_hextoRGB(self.grabbingColor)),1))
            painter.drawLine(16, 16, 20, 31)
            painter.drawLine(16, 16, 12, 31)
        if self.drag_state==STATE_ERASE:
            painter.setPen(QPen(QColor(*_hextoRGB(self.grabbingColor)),2))
            painter.drawPoint(16,16)
            painter.setPen(QPen(QColor(*_hextoRGB(self.grabbingColor)),1))
            painter.drawEllipse(QPoint(16,16),3,3)
        
        painter.end()

        # 将绘制好的矩形图标与原光标图标进行组合，并创建一个新的 QCursor 对象
        custom_cursor = QCursor(pixmap)

        # 将新的鼠标光标指针设置为应用程序的当前光标
        QGuiApplication.instance().setOverrideCursor(custom_cursor)


    #TODO 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 鼠标事件 
            #整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理#整理
        

        
    def mousePressEvent(self, event:QMouseEvent):
        xy = event.pos()
        xcl,ycl=xy.x(),xy.y()
        print(f'Clicked at position ({xcl}, {ycl})')
        print(f'aa({self.graphPos_(xcl,ycl)[0]}, {self.graphPos_(xcl,ycl)[1]})')

        def isMouseInScreen(mouseButton): 
            return event.button() == mouseButton and self.geometry().contains(xy)

        if self.drag_state==STATE_DRAG:
            if isMouseInScreen(Qt.MouseButton.LeftButton):
                self.flag_Dragging = True
                self.drag_start_pos = xy

            if isMouseInScreen(Qt.MouseButton.RightButton)and not self.flag_Dragging:
                self.flag_DraggingRight = True
                self.dragRight_start_pos = xy
        else:
            self.saved=False

            if isMouseInScreen(Qt.MouseButton.LeftButton):
                if self.drag_state==STATE_ERASE:self.lastdrawing=GrabitEraserLine(graghPos_=self.graphPos_)
                self.flag_Grabbing = True
                
            if isMouseInScreen(Qt.MouseButton.RightButton):
                self.flag_GrabbingRight = True

            color=self.colorSelector.getColor(xy)
            if color:
                self.grabbingColor=color
                self.colorSelector.setHighlight(color)

        
                

    def mouseMoveEvent(self, event:QMouseEvent):
        if self.flag_Dragging:
            self.dragRect = QRect(self.drag_start_pos, event.pos()).normalized()
            self.update()

        if self.flag_DraggingRight:
            
            new_pos=self.mousePos-self.dragRight_start_pos
            xshift,yshift=self.xshift+new_pos.x(),self.yshift+new_pos.y()
            self.pixmapRectTemp.setRect(xshift,yshift,int(self.q_image.width()*self.hscale), int(self.q_image.height()*self.vscale))
            self.update()
        
        if self.flag_Grabbing:
            #self.lastdrawing.append(self.mousePos)
            self.lastdrawing.fill(self.mousePos,event.pos())
            self.update()

        '''if self.keyStates.get(Qt.Key.Key_Delete)==KEY_HOLDING and self.colorSelector.getColor(self.mousePos) and self.colorSelector.getColor(self.mousePos)!=self.grabbingColor:
            self.colorSelector.remove(self.colorSelector.getColor(self.mousePos))
            self.update()'''

        #print(f"{self.flag_Dragging}")
        '''parent_x,parent_y=self.mapToParent(self.rect().topLeft()).x(),self.mapToParent(self.rect().topLeft()).y()
        xshift,yshift=self.xshift,self.yshift
        hscale,vscale=self.hscale,self.vscale'''
        
        '''self.graphPos_=lambda xcl,ycl:(
            ((xcl-xshift)/hscale-self.imgFigure.x1)/self.imgFigure.num_x*self.imgFigure.value_x+self.imgFigure.x1_value,
            ((ycl-yshift)/vscale-self.imgFigure.y1)/self.imgFigure.num_y*self.imgFigure.value_y+self.imgFigure.y1_value)
'''
        self.mousePos=event.pos()


    def mouseReleaseEvent(self, event:QMouseEvent):
        
        xy = event.pos()
        xcl,ycl=xy.x(),xy.y()
        print(f'Released at position ({xcl}, {ycl})')
        print(f'aa({self.graphPos_(xcl,ycl)[0]}, {self.graphPos_(xcl,ycl)[1]})')
        

        if event.button() == Qt.MouseButton.LeftButton and self.flag_Dragging:
            self.flag_Dragging = False
            x1,x2=(self.drag_start_pos.x()-self.xshift)/self.hscale,(xcl-self.xshift)/self.hscale
            y1,y2=(self.drag_start_pos.y()-self.yshift)/self.vscale,(ycl-self.yshift)/self.vscale
            self.imgFigure.recapImg(int(x1),int(x2),int(y1),int(y2))
            self.update()
            
        if event.button() == Qt.MouseButton.RightButton and self.flag_DraggingRight:
            self.flag_DraggingRight = False
            new_pos=self.mousePos-self.dragRight_start_pos
            self.xshift,self.yshift=self.xshift+new_pos.x(),self.yshift+new_pos.y()
            
            self.drag_start_pos,self.drag_release_pos=None,None
            self.dragRect=QRect()

            self.pixmapRect=QRect(self.pixmapRectTemp)
            self.commandStack.movePoints(new_pos.x(),new_pos.y())
            self.update()


        if event.button() == Qt.MouseButton.LeftButton and self.flag_Grabbing:
            
            if not self.flag_GrabbingRight:
                self.lastdrawing.color=self.grabbingColor
                print(self.grabbingColor)
                self.commandStack.push(self.lastdrawing)

            self.lastdrawing=GrabitLine(graghPos_=self.graphPos_)
            self.dataWidget.updateData(self.commandStack,self.graphPos_,self.getAlphaMap())
            self.flag_Grabbing=False
            self.flag_GrabbingRight = False
            self.update()

        if event.button() == Qt.MouseButton.RightButton and self.flag_GrabbingRight:
            self.flag_GrabbingRight = False
            self.commandStack.undo()
            self.dataWidget.updateData(self.commandStack,self.graphPos_,self.getAlphaMap())
            self.update()

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if ev.matches(QtGui.QKeySequence.StandardKey.Paste):
            clipboard = QApplication.clipboard()
            self.parseClipboard(clipboard=clipboard)
            
         
        self.keyStates[ev.key()] = KEY_HOLDING
        self.update()
        return super().keyPressEvent(ev)
    
    def parseClipboard(self,clipboard:QClipboard):
        if clipboard.mimeData().hasImage():
            image = clipboard.image()

    def keyReleaseEvent(self, a0: QKeyEvent | None) -> None:
         self.keyStates[a0.key()] = KEY_RELEASED
         self.update()
         return super().keyReleaseEvent(a0)
    
    
    '''def update(self):
         self.dataWidget.update()
         return super().update()'''

