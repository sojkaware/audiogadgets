# pip install pyqtchart
from PyQt5.QtWidgets import QMainWindow, QPlainTextEdit, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCharts import QtCharts
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal


class GuiTwoSubplots(QMainWindow):
    data_signal = pyqtSignal(QImage, list)

    def __init__(self):
        super().__init__()

        # create the chart and chart view for the line plot
        self.line_chart = QtCharts.QChart()
        self.line_chart_view = QtCharts.QChartView(self.line_chart)

        # create the chart and chart view for the spectrogram
        self.spec_chart = QtCharts.QChart()
        self.spec_chart_view = QtCharts.QChartView(self.spec_chart)

        # create the text area
        self.text_area = QPlainTextEdit()
        self.text_area.setMaximumBlockCount(3)

        # create the layout for the line plot and spectrogram
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.line_chart_view)
        plot_layout.addWidget(self.spec_chart_view)

        # create the layout for the entire window
        main_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout)
        main_layout.addWidget(self.text_area)

        # create a central widget and set the main layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # set a minimum size for the window
        self.setMinimumSize(800, 600)

        # connect the data signal to the receive_data method
        self.data_signal.connect(self.receive_data)

    def set_line_data(self, x_data, y_data):
        # clear any existing data from the chart
        self.line_chart.removeAllSeries()

        # create a new line series and add it to the chart
        series = QtCharts.QLineSeries()
        for x, y in zip(x_data, y_data):
            series.append(x, y)
        self.line_chart.addSeries(series)

        # create the X and Y axes and set their range
        axis_x = QtCharts.QValueAxis()
        axis_x.setRange(min(x_data), max(x_data))
        self.line_chart.addAxis(axis_x, QtCharts.Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QtCharts.QValueAxis()
        axis_y.setRange(min(y_data), max(y_data))
        self.line_chart.addAxis(axis_y, QtCharts.Qt.AlignLeft)
        series.attachAxis(axis_y)

        # set the title and axis labels
        self.line_chart.setTitle("Line Plot")
        axis_x.setTitleText("X Axis")
        axis_y.setTitleText("Y Axis")

        # update the chart view
        self.line_chart_view.setChart(self.line_chart)

    def set_spectrogram_data(self, data):
        # clear any existing data from the chart
        self.spec_chart.removeAllSeries()

        # create a new series and add it to the chart
        series = QtCharts.QLineSeries()
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                series.append(j, i, val)
        self.spec_chart.addSeries(series)

        # set the title and axis labels
        self.spec_chart.setTitle("Spectrogram")
        self.spec_chart.createDefaultAxes()

        # update the chart view
        self.spec_chart_view.setChart(self.spec_chart)

    def set_text(self, text):
        self.text_area.setPlainText(text)

    def receive_data(self, image, data):
        # set the image in the spectrogram chart
        pixmap = QPixmap.fromImage(image)
        self.spec_chart.setBackgroundBrush(pixmap)
        self.spec_chart.setBackgroundRoundness(0)
        self.spec_chart.setMargins(QMargins(0, 0, 0, 0))

        # set the line plot data
        self.set_line_data(range(len(data)), data)

    def update_data(self, image, data):
        self.data_signal.emit(image, data)