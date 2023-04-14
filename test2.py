import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from PyQt5.QtCore import QObject 
from pyqtgraph import ImageView

class GuiTwoSubplotsImageLinePlot(QMainWindow):
    def __init__(self):
        super().__init__()

        # Signal for sending data
        #self.data_signal = pyqtSignal(np.ndarray)

        # Set the title of the window
        self.setWindowTitle("PyQtChart Example")

        # Set the dimensions of the window
        self.setGeometry(100, 100, 800, 600)

        # Create a QVBoxLayout to hold the ImageView and chartView widgets
        widget = QWidget()
        layout = QVBoxLayout()

        # Create an ImageView widget to display the spectrogram image
        spectrogram = np.random.rand(200, 100)
        imageView = ImageView(self)
        imageView.setImage(spectrogram)

        # Add the ImageView and chartView widgets to the layout
        layout.addWidget(imageView)
        #layout.addStretch(1) # this glues it somehow
        layout.addSpacing(10) # this adds fixedspacing

        #self.add_image_to_layout_QImageQlabel(layout)
        self.add_lineplot_to_layout_QLineSeriesQChartView(layout)

        # Set the layout for the widget
        widget.setLayout(layout)

        # Set the widget as the central widget of the window
        self.setCentralWidget(widget)




    def add_image_to_layout_QImageQlabel(self, layout):
        # Create a QLabel with an image
        label = QLabel('Line plot')
        image = QImage('icon_mic.png')
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap)

        # Add the label to the layout
        layout.addWidget(label)


    def add_lineplot_to_layout_QLineSeriesQChartView(self, layout):
        # Create a QChart object
        chart = QChart()

        # Create a QLineSeries object
        series = QLineSeries()

        # Add data to the series
        series.append(0, 6)
        series.append(2, 4)
        series.append(3, 8)
        series.append(7, 4)
        series.append(10, 5)


        # Add the series to the chart
        chart.addSeries(series)

        # set the chart title and axis labels
        chart.setTitle("Data Plot")
        #chart.setAnimationOptions(QChart.SeriesAnimations) # this does the animation while resizing
        chart.createDefaultAxes()
        chart.axes()[0].setTitleText("X Axis")
        chart.axes()[1].setTitleText("Y Axis")

        # Create a QChartView object
        chartView = QChartView(chart)
  
        layout.addWidget(chartView)
        self.chart = chart
        self.chartView = chartView

    def update_plot(self, received_signal_data):
        # print(received_signal_data)
        # Create a QLineSeries for the FFT data
        series = QLineSeries()
        for i in range(len(received_signal_data)):
            series.append(i, received_signal_data[i])

        # Set the chart series and update the view
        self.chart.removeAllSeries()
        self.chart.addSeries(series)
        self.chart.createDefaultAxes()
        self.chartView.update()

# # Create an instance of QApplication
# app = QApplication(sys.argv)

# # Create an instance of Window
# window = GuiTwoSubplotsImageLinePlot()

# # Show the window
# window.show()

# # Run the event loop of QApplication
# sys.exit(app.exec_())