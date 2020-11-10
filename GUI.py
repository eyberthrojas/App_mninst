#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        physical_devices = tf.config.experimental.list_physical_devices('GPU') 
        for physical_device in physical_devices: 
            print(physical_device)
            tf.config.experimental.set_memory_growth(physical_device, True)
        tf.keras.backend.set_floatx('float64')
        self.title = 'GUI Demo AUTOMATAS'
        self.width = 1040
        self.height = 720
        self.left = 50
        self.top = 50 
        self.img = []
        self.model = keras.models.load_model('modelos/modelo1.h5')
        self.initUI()
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Widgets
        # Título 
        titulo = QtWidgets.QLabel(self)
        titulo.setAlignment(QtCore.Qt.AlignCenter)
        titulo.setStyleSheet('border: red; border-style:solid; border-width: 5px; color: black; font: 20pt Arial')
        titulo.setText("Predictor MNINST")
        # Grafica 
        self.canvas = Canvas(self, width = 8, height = 4)
        self.canvas.setStyleSheet('border: blue; border-style:solid; border-width: 5px')
        # Boton Inicio
        self.boton_predecir = QtWidgets.QPushButton(self)
        self.boton_predecir.setText("Predecir digito")
        self.boton_predecir.setStyleSheet('border: red; border-style:solid; border-width: 5px; color: blue; font: 15pt Arial')
        self.boton_predecir.clicked.connect(self.predecir)
        # Visualización
        self.prediccion = QtWidgets.QLabel(self)
        self.prediccion.setAlignment(QtCore.Qt.AlignCenter)
        self.prediccion.setStyleSheet('border: gray; border-style:solid; border-width: 1px; color: black; font: 20pt Arial')
        
        # Layouts
        self.controles = QtWidgets.QHBoxLayout(self)
        self.controles.addWidget(self.boton_predecir,30)
        self.controles.addWidget(self.prediccion,70)
        # Crea layoud raiz
        self.raiz = QtWidgets.QVBoxLayout(self)
        self.raiz.addWidget(titulo,5)
        self.raiz.addWidget(self.canvas,70)
        self.raiz.addLayout(self.controles,25)
        # main
        self.main = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main)
        self.centralWidget().setLayout(self.raiz)
        self.show()
    def predecir(self):
        ima_test = cv.imread('imagen.png',2)
        ima_resize = cv.resize(ima_test,(20,20))/255
        self.canvas.plot(ima_resize)
        ima_resize = ima_resize.T
        ima_reshape = ima_resize.reshape([1,400])
        pred = np.argmax(self.model.predict(ima_reshape,verbose=0))
        self.prediccion.setText("El digito escrito correspone a un {}".format(str(pred)))
        
    
class Canvas(FigureCanvas):
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
    def plot(self, ima):
        self.axes.cla()
        self.axes.imshow(ima, cmap = 'gray')
        self.draw()


        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
