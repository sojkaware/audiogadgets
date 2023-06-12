import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class GuiDebugTwoCols(tk.Frame):

    def __init__(self, master=tk.Tk(), **kwargs):
        super().__init__(master, **kwargs)
        self.root = master


        self.left_col = self.create_column()
        self.left_col.grid(row=0, column=0, sticky="nsew")

        self.right_col = self.create_column()
        self.right_col.grid(row=0, column=1, sticky="nsew")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)



    def run_standalone(self, title='Main window'):
        #root = tk.Tk()
        self.root.title(title)
        self.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        self.root.mainloop()

    def print_something(self):
        print("shit")
       

    def create_column(self):
        col = tk.Frame(self)

        mydpi = 50

        # Top section
        top_frame = tk.Frame(col)
        top_frame.pack(expand=True, fill=tk.BOTH)
        fig = Figure(figsize=(5, 8), dpi=mydpi)
        self.ax_img = fig.add_subplot(111)
        self.canvas_img = FigureCanvasTkAgg(fig, master=top_frame)
        self.canvas_img.draw()
        self.canvas_img.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Middle sections
        middle_frame = tk.Frame(col)
        middle_frame.pack(expand=True, fill=tk.BOTH)

        fig1 = Figure(figsize=(5, 4), dpi=mydpi)
        self.ax_line1 = fig1.add_subplot(111)
        self.canvas_line1 = FigureCanvasTkAgg(fig1, master=middle_frame)

        #self.plot3.clear()
        data = np.array([1, 2, 3, 4])
        self.ax_line1.plot(data)
        self.ax_line1.set_title("Data Plot")
        
        self.canvas_line1.draw()
        self.canvas_line1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig2 = Figure(figsize=(5, 4), dpi=mydpi)
        self.ax_line2 = fig2.add_subplot(111)
        self.canvas_line2 = FigureCanvasTkAgg(fig2, master=middle_frame)
        self.canvas_line2.draw()
        self.canvas_line2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bottom section
        bottom_frame = tk.Frame(col)
        bottom_frame.pack(expand=True, fill=tk.BOTH)
        self.text_area = tk.Text(bottom_frame, height=3)
        self.text_area.pack(side=tk.BOTTOM, fill=tk.BOTH)

        return col



class GuiTwoBtnApp(tk.Frame):
    timeSin = 0.0

    def __init__(self, master=None, debug_gui=None, **kwargs):
        super().__init__(master, **kwargs)
        self.debug_gui = debug_gui # just for debugging

        self.left_btn = tk.Button(self, text="Left", command=self.left_btn_click)
        self.left_btn.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.right_btn = tk.Button(self, text="Right", command=self.right_btn_click)
        self.right_btn.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    def left_btn_click(self):
        # Generate data for the first plot
        data1 = np.random.rand(100, 50)
        self.debug_gui.update_plot(col=0, plot=0, data=data1)

        # Generate data for the second plot
        data2 = np.random.rand(100)
        self.debug_gui.update_plot(col=0, plot=1, data=data2)

        # Generate data for the third plot
        self.timeSin = self.timeSin + 0.001
        data3 = np.sin(np.linspace(0, 2*np.pi*self.timeSin, 100))
        self.debug_gui.update_plot_bysample(col=0, plot=2, data=data3)

        # Generate data for the text area
        text_data = 'Random Text: ' + str(np.random.randint(100))
        self.debug_gui.update_text_byline(col=0, text_area=0, data=text_data)
        
    def right_btn_click(self):
        print( "to be implemented, but later" )

       


if __name__ == "__main__":

    #root = tk.Tk()
    #root.title("App")

    # btn_app = GuiTwoBtnApp()
    

    debug_gui = GuiDebugTwoCols()
    debug_gui.run_standalone(title = 'Debug visualization app')
    

    # btn_app.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)

    # root.mainloop()

    