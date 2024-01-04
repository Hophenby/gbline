

from PyQt6.QtGui import QImage,QPixmap,QPen,QPainter,QColor,QMouseEvent,QPaintEvent
from PyQt6.QtCore import Qt,QRect,QPoint
import numpy as np

from ColorSelector import _hextoRGB, _rgbtoHEX

class Node:
    def __init__(self, data:QPoint|None=None, graphdata:tuple[int,float]=None):
        
        self.data = data
        self.graphdata=graphdata
        graphdata=(int(graphdata[0]),graphdata[1])
        self.next = None
        self.erased=False
    
    def movePoint(self,d_x,d_y):
        self.data=self.data+QPoint(d_x,d_y)
    
class GrabitLine:
    '''神秘画线为什么我要用这种来存数据为什么我要用这种来存数据为什么我要用这种来存数据为什么我要用这种来存数据哇啊啊啊啊哇哇啊啊啊啊'''
    def __init__(self,line:dict[int,int]|None=None,color=None,graghPos_=None,parentPos_=None):
        self.toolmode=0
        self.head = None
        self.graphPos_=graghPos_
        self.parentPos_=parentPos_
        self.color=color
        self.info=None
        self.current_iter=None
        if not (line is None):
            for x,y in line.items():
                self.append(QPoint(x,y))


    def __iter__(self):
        self.current_iter=self.head
        return self
    
    def __next__(self):
        if self.current_iter:
            pdata=self.current_iter.data
            self.current_iter=self.current_iter.next
            return pdata
        raise StopIteration

    def isEmpty(self):
        return self.head is None
    
    def lastnode(self):
        if self.head is None:
            return
        else:
            current = self.head
            while current.next:
                current = current.next
            return current
        
    def fill(self,p0:QPoint,p1=None,erased=False):
        if self.lastnode() is None: 
            self.append(data=p0,graphdata=self.graphPos_(p0.x(),p0.y()),erased=erased)
        if p1 is None: p1=self.lastnode().data
        gdata0,gdata1=self.graphPos_(p0.x(),p0.y()),self.graphPos_(p1.x(),p1.y())
        x=np.linspace(p0.x(),p1.x(),int(np.abs(gdata0[0]-gdata1[0])+1),dtype=np.int32)
        y=np.linspace(p0.y(),p1.y(),int(np.abs(gdata0[0]-gdata1[0])+1),dtype=np.int32)
        gx=np.linspace(gdata0[0],gdata1[0],int(np.abs(gdata0[0]-gdata1[0])+1),dtype=np.int32)
        gy=np.linspace(gdata0[1],gdata1[1],int(np.abs(gdata0[0]-gdata1[0])+1),dtype=np.float32)
        for xp,yp,xg,yg in zip(x,y,gx,gy):
            self.append(data=QPoint(xp,yp),graphdata=(xg,yg),erased=erased)
        

    def append(self, data:QPoint,graphdata:tuple[int,float]|None=None,erased=False):
        if graphdata is None:graphdata=self.graphPos_(data.x(),data.y())
        new_node = Node(data=data,graphdata=graphdata)
        new_node.erased=erased
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
                if int(graphdata[0])==int(current.graphdata[0]):
                    current.data=new_node.data
                    current.graphdata=new_node.graphdata
                    
            '''    
            if data.x()-current.data.x()>1:
                self.append(QPoint(int(current.data.x()+1),
                                   int(current.data.y()+(data.y()-current.data.y())/(data.x()-current.data.x()))))
            if data.x()-current.data.x()<-1:
                self.append(QPoint(int(current.data.x()-1),
                                   int(current.data.y()-(data.y()-current.data.y())/(data.x()-current.data.x()))))'''
            '''if np.abs(graphdata[0]-current.graphdata[0])>1:
                self.append(data=QPoint(int((data.x()+current.data.x())/2),
                                        int((data.y()+current.data.y())/2)),
                            graphdata=((graphdata[0]+current.graphdata[0])/2,
                                       (graphdata[1]+current.graphdata[1])/2),
                            erased=erased)'''
            current.next = new_node
            
    def findBound(self):
        if self.isEmpty(): return
        min_x,max_x,min_y,max_y=self.head.data.x(),self.head.data.x(),self.head.data.y(),self.head.data.y()
        for pdata in self:
            min_x=min(min_x,pdata.x())
            min_y=min(min_y,pdata.y())
            max_x=max(max_x,pdata.x())
            max_y=max(max_y,pdata.y())
        return min_x,max_x,min_y,max_y

    def movePoints(self,d_x,d_y):
        current = self.head
        while current:
            current.movePoint(d_x,d_y)
            current = current.next

    def display(self,painter:QPainter,condition=None):
        if condition is None:   condition=lambda x:True
        current = self.head
        while current:
            if condition(current.data):
                if current.next:
                    painter.drawLine(current.data,current.next.data)
                #painter.setPen()
                painter.drawPoint(current.data)
            current = current.next



class GrabitEraserLine(GrabitLine):     #神秘橡皮擦
    def __init__(self, line=None, color=None,graghPos_=None,parentPos_=None):
        super().__init__(line=line, color=color,graghPos_=graghPos_,parentPos_=parentPos_)
        self.toolmode=1
        
        if not (line is None):
            self.head=None
            for x,y in line.items():
                self.append(QPoint(x,y))

    def append(self, data: QPoint,graphdata=None,erased=True):
        return super().append(data,graphdata=graphdata,erased=erased)
    
    def fill(self, p0: QPoint, p1=None, erased=True):
        return super().fill(p0, p1, erased)

    def display(self, painter: QPainter, condition=None):
        if self.isEmpty():return
        #painter.fillRect(QRect(self.head.data,self.lastnode().data))
        if condition is None:   condition=lambda x:True
        _,_,y1,y2=self.findBound()
        current = self.head
        #p0=None
        while current:
            #p0=p0 or current
            if condition(current.data):
                #painter.setPen(QPen(QColor(155,155,155,20)))
                if current.next:
                    painter.fillRect(QRect(QPoint(current.data.x(),y1),QPoint(current.next.data.x(),y2)),QColor(225,225,225,20))
                #painter.drawPoint(current.data)
            '''    pass
            else:
                painter.fillRect(QRect(p0.data,current.data),QColor(155,155,155,10))
                p0=current'''
                
            current = current.next
        return #super().display(painter, condition)

class GrabitStack:
    '''
    命令堆栈用来存操作
    ## 哇啊啊啊啊哇啊啊啊
    ## 哇啊啊啊啊哇啊啊啊
    ## 哇啊啊啊啊哇啊啊啊
    '''
    def __init__(self,*args):
        self.stack = []+list(*args)
        self.cancelled=[]

    def __iter__(self):
        return iter(self.stack)

    def raw_push(self, item:GrabitEraserLine|GrabitLine):
        if item.__class__==GrabitEraserLine and self.isEmpty():return
        if self.findInfoFromColor(item.color): 
            item.info=self.findInfoFromColor(item.color)
            print(item.info)
        self.stack.append(item)

    def push(self, item):
        self.raw_push(item)
        self.cancelled.clear()

    def pop(self):
        if not self.isEmpty():
            return self.stack.pop()

    def isEmpty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)
    
    def undo(self):
        last=self.pop()
        self.cancelled.append(last)

    def redo(self):
        if len(self.cancelled) == 0: return
        last=self.cancelled.pop()
        self.raw_push(last)

    def movePoints(self,d_x,d_y):
        for element in self.stack:
            element.movePoints(d_x,d_y)
    
    def findInfoFromColor(self,color):
        for element in self.stack:
            if element.color==color:
                return element.info
            
    
    def setInfoFromColor(self,color,info):
        for element in self.stack:
            if element.color==color:
                element.info=info
            
    

    def drawCasting(self,painter:QPainter,colorAlphaMap:dict={}):
        painted={}
        pen=painter.pen()
        pen.setStyle(Qt.PenStyle.SolidLine)
        for element in reversed(self.stack):
            color=element.color or Qt.GlobalColor.black
            if type(color)==type(""): color=_hextoRGB(color)
            painted.setdefault(color,set())

            defaultAlpha=colorAlphaMap.get("default",255)
            pen.setColor(QColor(*color,colorAlphaMap.get(_rgbtoHEX(color),defaultAlpha)))
            #print(colorAlphaMap.get(color,defaultAlpha))

            pen.setWidth(2)
            painter.setPen(pen)
            element.display(painter,lambda pos:not (pos.x() in painted[color]))

            if element.findBound():
                minx,maxx,_,_=element.findBound()
                painted[color].update(set(range(minx,maxx+1)))

        #哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊哇哇哇哇哇哇哇啊啊啊啊啊啊啊啊啊啊
        pass
        
