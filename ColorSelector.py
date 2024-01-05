
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QApplication,  QWidget, QLabel,QPushButton,QTableWidget,QTableWidgetItem
from PyQt6.QtGui import QBrush, QImage, QPainterPath,QPixmap,QPen,QPainter,QColor,QMouseEvent,QPaintEvent
from PyQt6.QtCore import QPointF, Qt,QRect,QPoint

def _rgbtoHEX(r,g=None,b=None):
    '''出来的是
    # #aaaaaa
    这种'''
    if not (g and b):r,g,b=r
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def _hextoRGB(hex:str) -> tuple[int,int,int]:
    '''出来的是
    `tuple(r,g,b)`
    这种'''
    r, g, b = int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)
    return (r,g,b)

def _qcolortoHEX(qc:QColor|Qt.GlobalColor):
    return _rgbtoHEX((QColor(qc).getRgb()[:3]))

def _hextoQColor(hex:str):
    return QColor(*_hextoRGB(hex))

class ColorSelector(QLabel):
    def __init__(self,parent,colorRadius=10,column_mode=False):
        super().__init__(parent)
        self.COLORS=[color for color in map(_qcolortoHEX,
                    (Qt.GlobalColor.black,))]
        self.additionalcolor=[color for color in map(_qcolortoHEX,
                    (Qt.GlobalColor.red,Qt.GlobalColor.yellow,
                     Qt.GlobalColor.blue,Qt.GlobalColor.green,
                     Qt.GlobalColor.cyan,Qt.GlobalColor.magenta,
                     Qt.GlobalColor.darkRed,Qt.GlobalColor.darkYellow,
                     Qt.GlobalColor.darkBlue,Qt.GlobalColor.darkGreen,
                     Qt.GlobalColor.darkCyan,Qt.GlobalColor.darkMagenta))]
        self.colomnmode=column_mode
        self.customColors=[]
        self.displayColors=self.COLORS+self.customColors            #self.displayColors=self.COLORS+self.customColors
        self.radius=colorRadius
        self.highlightColors=set()

    def __contains__(self,color):
        return color in self.displayColors
    
    def __len__(self):
        return len(self.displayColors)
    
    def appendNew(self):
        for c in self.additionalcolor:
            if c in self.displayColors:continue
            self.append(c)
            return

    def append(self,*args:str):
        for hex in set(args):
            self.customColors.append(hex)
            self.displayColors=self.COLORS+self.customColors
            self.update()

    def remove(self,*args:str):
        for hex in args:
            if hex in self.customColors:
                self.customColors.remove(hex)
            self.displayColors=self.COLORS+self.customColors
            self.update()

    def setHighlight(self,colors:str|set[str]):
        if type(colors)==type(""):
            self.highlightColors={colors}
            self.update()
            return
        self.highlightColors=colors
        self.update()

    def getColor(self,pos:QPoint):
        if self.colomnmode: return self.getColorColomn(pos)
        return self.getColorRow(pos)

    def getColorRow(self,pos:QPoint):
        geo=self.geometry()
        if pos in geo:
            colorIndex= (len(self.displayColors)*(pos.x()-geo.left())/geo.width())//1
            return self.displayColors[int(colorIndex)]
        
    def getColorColomn(self,pos:QPoint):
        geo=self.geometry()
        if pos in geo:
            colorIndex= (len(self.displayColors)*(pos.y()-geo.top())/geo.height())//1
            return self.displayColors[int(colorIndex)]
    
    def getColorRect(self,gcolor:Qt.GlobalColor|QColor|str):
        if self.colomnmode: return self.getColorRectColomn(gcolor)
        return self.getColorRectRow(gcolor)

    def getColorRectRow(self,gcolor:Qt.GlobalColor|QColor|str):
        if type(gcolor)!=type("   ") :
            gcolor=_qcolortoHEX(gcolor)
        if not (gcolor in self.displayColors):return
        colorIndex=self.displayColors.index(gcolor)
        geo=self.geometry()
        return QRect(
            int(geo.left()+colorIndex/len(self.displayColors)*geo.width()),
            geo.top(),
            int(geo.width()/len(self.displayColors)),
            geo.height()
            )
    
    def getColorRectColomn(self,gcolor:Qt.GlobalColor|QColor|str):
        if type(gcolor)!=type("   ") :
            gcolor=_qcolortoHEX(gcolor)
        if not (gcolor in self.displayColors):return
        colorIndex=self.displayColors.index(gcolor)
        geo=self.geometry()
        return QRect(
            geo.left(),
            int(geo.top()+colorIndex/len(self.displayColors)*geo.height()),
            geo.width(),
            int(geo.height()/len(self.displayColors))
            )

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter=QPainter(self)
        pen=QPen(Qt.GlobalColor.lightGray,3)
        painter.setPen(pen)
        painter.fillRect(self.rect(),QColor(255,255,255))
        painter.drawRect(self.rect())
        for color in self.displayColors:
            pen=QPen(QColor(*_hextoRGB(color)),3)
            painter.setPen(pen)
            r=self.radius
            painter.drawEllipse(self.mapFromParent(self.getColorRect(color).center()),r,r)
            if color in self.highlightColors:
                path = QPainterPath()
                path.addEllipse(QPointF(self.mapFromParent(self.getColorRect(color).center())),r,r)
                brush = QBrush(QColor(*_hextoRGB(color)) , Qt.BrushStyle.SolidPattern)
                painter.fillPath(path, brush)
                
        return super().paintEvent(a0)


