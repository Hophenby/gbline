import sys
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QSize
from PyQt6.QtWidgets import QApplication, QWidget, QLineEdit, QHBoxLayout, QMainWindow, QSizePolicy
from PyQt6.QtGui import QContextMenuEvent, QCursor, QFont, QGuiApplication, QIcon, QImage, QKeyEvent, QMouseEvent, QPaintEvent, QShowEvent, QPainter, QPen, QBrush

class DoubleLineEdit(QWidget):
    def __init__(self, parent:QWidget=None):
        super(DoubleLineEdit, self).__init__(parent)

        # 创建两个输入框
        fixedsize=QSize(70,20)
        self.setFixedSize(fixedsize)
        self.left_edit = QLineEdit()
        #self.left_edit.setGeometry(fixedsize)
        self.right_edit = QLineEdit()
        #self.left_edit.setGeometry(fixedsize)

        self.saved=[]
        self.showpos=None

        # 将两个输入框放在水平布局中
        layout = QHBoxLayout()
        self.left_edit.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        self.right_edit.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        layout.addWidget(self.left_edit)
        layout.addWidget(self.right_edit)
        layout.setContentsMargins(2, 2, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)

        self.connectedslot=None

        # 隐藏输入框
        self.hide()

    def showEvent(self, a0: QShowEvent | None) -> None:
        # 设置输入框位置为鼠标位置
        # print("ww")
        
        pos = self.parent().mapFromGlobal(self.cursor().pos())
        self.move(pos)
        self.showpos=pos
        # 显示输入框
        self.show()
        self.left_edit.setFocus()
        return super().showEvent(a0)
    
    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter=QPainter(self)
        if self.showpos:
            #print("drawing point")
            painter.setPen(QPen(Qt.GlobalColor.blue,5))
            painter.drawPoint(QPoint(0,0))
        return super().paintEvent(a0)
    
    
    '''def contextMenuEvent(self, a0: QContextMenuEvent | None) -> None:
        # 设置输入框位置为鼠标位置
        #print("ww")
        if a0.reason()==a0.Type.MouseButtonPress:
            pos = self.parent().mapFromGlobal(self.cursor().pos())
            self.move(pos)

            # 显示输入框
            self.show()
            self.left_edit.setFocus()
        return super().contextMenuEvent(a0)'''

    def hideEvent(self, event):
        # 隐藏输入框并重置内容
        self.hide()
        self.left_edit.setText('')
        self.right_edit.setText('')

    def keyPressEvent(self, event:QKeyEvent):
        if event.key() == Qt.Key.Key_Return:
            # 返回两个输入框的内容并隐藏输入框
            left_text = self.left_edit.text()
            right_text = self.right_edit.text()
            print(f"recording pos ({left_text}, {right_text})")
            try:
                assert  not(self.showpos is None)
                self.append((QPoint(self.showpos),(float(left_text),float(right_text))))
            except Exception as e:
                print(f"failed to record pos({left_text}, {right_text})")
                print(f"[{e.__class__.__name__}] {e}")
            self.hide()
            print(f"saved data: {self.saved}")
            '''elif event.key() == Qt.Key.Key_Tab:
            print("left edit focus set")
            # 切换焦点到左侧输入框
            self.left_edit.setFocus()
        elif event.key() == Qt.Key.Key_Right:
            print("right edit focus set")
            # 切换焦点到右侧输入框
            self.right_edit.setFocus()'''
        else:
            super(DoubleLineEdit, self).keyPressEvent(event)

    def connectRemapFunction(self,slot):
        self.connectedslot=slot

    def showAtPos(self,globalPosition:QPointF):
        self.move(QPointF.toPoint(globalPosition))
        self.show()

    def append(self,item):
        self.saved.append(item)
        if len(self.saved)>=3:
            self.saved.pop(0)
        if len(self.saved)==2 and not (self.connectedslot is None):
            p1,(x1v,y1v)=item
            p2,(x2v,y2v)=self.saved[0]
            #print(p1,(x1v,y1v),p2,(x2v,y2v))
            self.connectedslot(p1,p2,(x1v,y1v),(x2v,y2v))
            print("New map from graph updated")


'''class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # 创建 DoubleLineEdit 控件，并设置为 centralWidget
        self.double_edit = DoubleLineEdit()
        self.setCentralWidget(self.double_edit)

    def mousePressEvent(self, event:QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            # 右键点击时显示 DoubleLineEdit 控件
            pos = self.mapFromGlobal(event.globalPosition())
            self.double_edit.showAtPos(pos)'''
'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window=QMainWindow()
    demo = DoubleLineEdit(window)
    window.setCentralWidget(demo)
    window.setMouseTracking(True)
    window.show()
    sys.exit(app.exec())
'''
'''if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())'''