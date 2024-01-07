import sys
from PyQt6.QtWidgets import  QApplication
#from qasync import QApplication,QEventLoop
#import asyncio
from interactiveplot import InteractivePlot


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = InteractivePlot()
    mainWindow.show()
    sys.exit(app.exec())