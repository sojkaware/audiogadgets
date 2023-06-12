
import sys
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from data_visualizer import DataVisualizer


class DataGenerator(QMainWindow):
    data_ready = pyqtSignal(np.ndarray, np.ndarray, float, str)

    def __init__(self):
        super().__init__()

        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 100 ms
        self.timer.timeout.connect(self.generate_data)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Data Generator')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def generate_data(self):
        matrix = np.random.rand(10, 10)
        vector = np.random.rand(10)
        sinus_sample = np.sin(np.random.rand())
        text = "Random value: " + str(np.random.rand())

        self.data_ready.emit(matrix, vector, sinus_sample, text)

    def start(self):
        self.timer.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_gen = DataGenerator()
    data_viz = DataVisualizer()

    data_gen.data_ready.connect(data_viz.update_data)

    data_gen.show()
    data_viz.show()

    data_gen.start()

    sys.exit(app.exec_())


