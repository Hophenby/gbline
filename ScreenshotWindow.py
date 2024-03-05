#import asyncio
from PyQt6.QtWidgets import QDialog, QLabel, QApplication, QVBoxLayout
from PyQt6.QtGui import QPaintEngine, QPixmap, QPainter, QPen, QColor, QPainterPath, QMouseEvent, QPaintEvent, QKeyEvent,QBrush,QCursor
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal, QThread, QWaitCondition, QMutex,QPointF

from enum import Enum


class CaptureThread(QThread):
    capture_finished = pyqtSignal(object)

    def __init__(self, screenshot_dialog,status):
        super().__init__()
        self.screenshot_dialog = screenshot_dialog
        self.status=status

    def run(self):
        result = self.screenshot_dialog.capture(self.status)
        if result:
            self.capture_finished.emit(result)




class ScreenshotDialog(QDialog):
    class state(Enum):
        capture=0
        color_select=1

    def __init__(self, parent=None):
        super().__init__(parent)

        # 创建标签用于显示截图
        self.ScreenshotLabel = ScreenshotLabel(self)
        self.status=None
        #if status: self.status=status

        #self.capture_event = asyncio.Event()
        self.captured=None
        self.loop=None
        self.wait_condition = QWaitCondition()
        self.mutex = QMutex()
        
        # 设置窗口属性
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 创建垂直布局并将标签添加到布局中
        layout = QVBoxLayout(self)
        layout.addWidget(self.ScreenshotLabel)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        # 记录鼠标按下和释放的坐标
        self.start_pos = None
        self.end_pos = None

        # 获取当前屏幕截图
        screen = QApplication.screenAt(QCursor.pos())
        self.screenshot = screen.grabWindow(0)
        self.ScreenshotLabel.setPixmap(self.screenshot)
        self.setMouseTracking(True)
        self.showFullScreen()

    def capture(self,status):
        # 设置截图
        # 获取当前屏幕截图
        self.status=status
        self.ScreenshotLabel.setPixmap(self.screenshot)
        self.setMouseTracking(True)
        self.showFullScreen()
        print("waiting for selection")
        #self.loop = qasync.QEventLoop()
        self.mutex.lock()
        self.wait_condition.wait(self.mutex)
        self.mutex.unlock()
        return self.captured

    '''def color_select(self):
        # 设置截图
        # 获取当前屏幕截图
        self.status=self.state.color_select
        self.ScreenshotLabel.setPixmap(self.screenshot)
        self.setMouseTracking(True)
        self.showFullScreen()
        print("waiting for selection")
        self.mutex.lock()
        self.wait_condition.wait(self.mutex)
        self.mutex.unlock()
        return self.captured'''

    def mousePressEvent(self, event: QMouseEvent):
        if self.status==self.state.capture:
            self.ScreenshotLabel.start_pos = event.pos()


    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        self.ScreenshotLabel.end_pos = event.pos()
        self.update()
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.ScreenshotLabel.mousePos=event.pos()
        if self.status==self.state.capture:
            self.ScreenshotLabel.end_pos = event.pos()

            # 进行截图
            screenshot_rect = QRect(self.ScreenshotLabel.start_pos, self.ScreenshotLabel.end_pos).normalized()
            screenshot = self.screenshot.copy(screenshot_rect)
            #screenshot.save("screenshot.png")

            # 返回QPixmap对象
            self.captured=(screenshot)
            print(type(self.captured))
            self.wait_condition.wakeAll()
            self.close()


        if self.status==self.state.color_select:
            x,y=event.pos().x(), event.pos().y()
            image=self.screenshot.toImage()
            w,h=image.width(),image.height()
            w1,h1=self.width(),self.height()
            xf,yf=int(x/w1*w),int(y/h1*h)
            self.captured=self.screenshot.toImage().pixelColor(xf,yf)
            print(type(self.captured))
            self.wait_condition.wakeAll()
            self.close()
            '''self.capture_event.set()
            if self.loop:   self.loop.stop()'''
            #self.wait_condition.wakeAll()

'''
    def return_pixmap(self, pixmap):
        # 在这里处理返回的QPixmap对象
        # print("返回截图QPixmap对象")
        pass'''

'''class CaptureThread(QThread):
    capture_finished = pyqtSignal(object)

    def __init__(self, screenshot_dialog:ScreenshotDialog):
        super().__init__()
        self.screenshot_dialog = screenshot_dialog

    def run(self):
        result = self.screenshot_dialog.capture()
        self.capture_finished.emit(result)'''

class ScreenshotLabel(QLabel):
    def __init__(self,parent) -> None:
        super().__init__(parent=parent)
        self.start_pos = None
        self.end_pos = None
        
        self.mousePos = None
        self.setMouseTracking(True)

    def mouseMoveEvent(self, ev: QMouseEvent | None) -> None:
        self.end_pos = ev.pos()
        return super().mouseMoveEvent(ev)

    def paintEvent(self, a0: QPaintEvent | None) -> None: 
        super().paintEvent(a0)
        painter = QPainter(self)
        
        painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
        painter.drawRect(self.rect())
        #print(self.end_pos)
        if self.end_pos:
            mousePos=(self.end_pos)
            x,y=mousePos.x(), mousePos.y()
            path = QPainterPath()
            path.addEllipse(QPointF(x+20,y+20),16,16)
            image=self.pixmap().toImage()
            w,h=image.width(),image.height()
            w1,h1=self.width(),self.height()
            xf,yf=int(x/w1*w),int(y/h1*h)
            brush = QBrush(QColor(image.pixel(xf,yf)) , Qt.BrushStyle.SolidPattern)
            painter.fillPath(path, brush)
            painter.setPen(QPen(QColor(0,0,0),4))
            painter.drawEllipse(QPointF(x+20,y+20),16,16)
            #painter.drawText(self.end_pos-QPoint(50,50),f"{w}:{h}-({x}, {y})-{w1}:{h1}")

        if self.start_pos and self.end_pos:
            #print("1111")
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
            painter.drawRect(QRect(self.start_pos, self.end_pos))

        return