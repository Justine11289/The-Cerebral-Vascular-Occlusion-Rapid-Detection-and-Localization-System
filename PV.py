from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QSlider
from PyQt5.QtWidgets import QColorDialog, QMessageBox, QLabel
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

class PointCloudViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.current_file_path = None
        self.current_actor = None
        self.point_cloud_list = []
        self.current_file_paths = []
        self.initUI()
        self.show()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Button style
        button_style = "QPushButton { background-color: #4CAF50; color: white; border: 1px solid gray; border-radius: 4px; padding: 12px; font-size: 20px; }"
        clear_button_style = "QPushButton { background-color: #f44336; color: white; border: 1px solid gray; border-radius: 4px; padding: 12px; font-size: 20px; }"

        # Row 1: Buttons
        button_row_layout = QHBoxLayout()

        # Open point cloud file
        self.open_button = QPushButton("打開點雲文件")
        self.open_button.setStyleSheet(button_style)
        self.open_button.clicked.connect(self.openPointCloud)
        button_row_layout.addWidget(self.open_button)


        # Clear point cloud files
        self.clear_button = QPushButton("清除點雲")
        self.clear_button.setStyleSheet(clear_button_style)
        self.clear_button.clicked.connect(self.clearPointCloud)
        button_row_layout.addWidget(self.clear_button)

        self.layout.addLayout(button_row_layout)

        # Row 2: Slider
        self.point_size_layout = QHBoxLayout()
        self.layout.addLayout(self.point_size_layout)

        label_text = QLabel("點雲大小")
        self.point_size_layout.addWidget(label_text)

        self.point_size_slider = QSlider()
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(20)
        self.point_size_slider.setValue(5)
        self.point_size_slider.setOrientation(1)
        self.point_size_slider.valueChanged.connect(self.updatePointSize)
        self.point_size_layout.addWidget(self.point_size_slider)

        # Display the current value of the slider
        self.point_size_value_label = QLabel(f"({self.point_size_slider.value()})")
        self.point_size_layout.addWidget(self.point_size_value_label)
        self.point_size_slider.valueChanged.connect(self.updatePointSizeLabel)


        # Row 3: Viewer
        self.viewer = BackgroundPlotter(window_size=(800, 600))
        self.viewer.set_background("white")
        self.layout.addWidget(self.viewer.interactor)

        # Row 4: Display opened files
        self.file_info_label = QLabel("已開啟：")
        self.layout.addWidget(self.file_info_label)

    def openPointCloud(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "打開點雲文件", "", "點雲文件 (*.pts *ply);;所有文件 (*)", options=options)

        if file_name:
            color = self.chooseColor()
            self.loadPointCloud(file_name, color)

    def clearPointCloud(self):
        self.viewer.clear()
        print("Pointcloud cleared")
        self.updateFileInfoLabel()

    def chooseColor(self):
        color_dialog = QColorDialog(self)
        color_dialog.setWindowTitle("選擇點雲顏色")
        color = color_dialog.getColor()
        return color.getRgbF() if color.isValid() else (1.0, 0.0, 0.0)

    def loadPointCloud(self, file_path, color):
        try:
            # Load point cloud
            point_cloud = pv.read(file_path)
            print(f"Loaded {len(point_cloud.points)} points from {file_path}")

            actor = self.viewer.add_mesh(point_cloud, render_points_as_spheres=True, point_size=self.point_size_slider.value(), color=color)
            self.point_cloud_list.append(actor)

            self.current_file_path = file_path
            self.current_file_paths.append(file_path)
            self.viewer.show()

            self.updateFileInfoLabel()

        except Exception as e:
            print(f"Error loading or visualizing point cloud: {str(e)}")

    def updatePointSize(self):
        point_size = self.point_size_slider.value()
        print(f"更新點的大小：{point_size}")

        for actor in self.point_cloud_list:
            actor.GetProperty().SetPointSize(point_size)

        self.viewer.render()

    def updatePointSizeLabel(self):
        value = self.point_size_slider.value()
        self.point_size_value_label.setText(f"({value})")

    def updateFileInfoLabel(self):
        # Display the list of opened files
        opened_files = [os.path.basename(file_path) for file_path in self.current_file_paths]
        print(opened_files)
        self.file_info_label.setText("已開啟檔案：" + " ; ".join(opened_files))

def run_tkinter():
    root = tk.Tk()
    root.title("The-Cerebral-Vascular-Occlusion-Rapid-Detection-and-Localization-System")


    # Adding a title label with improved styling
    title_label = tk.Label(root, text="腦血管阻塞快速檢測與定位系統", font=("Helvetica", 20, "bold"))
    title_label.pack(side=tk.TOP, pady=(20, 10))  # Adjusted padding for better spacing

    # Label to display the selected file name
    selected_file_label = tk.Label(root, text="", font=("Helvetica", 12), wraplength=200)
    selected_file_label.pack(side=tk.TOP, pady=(0, 10))

    def process_and_run_final_pts():
        global input_file
        input_file = filedialog.askopenfilename(filetypes=[("NIfTI File", "*.nii.gz")])
        if input_file:
            # Update the label with the selected file name
            selected_file_label.config(text=f"選擇檔案: {os.path.basename(input_file)}")

            # Adjust the button magnification and make it in the middle of the UI
            process_button.config(state=tk.DISABLED)  # Disable the button during processing

            # Add progress bar
            progress_bar = ttk.Progressbar(root, mode='indeterminate')
            progress_bar.pack(side=tk.TOP, anchor=tk.CENTER, pady=(0, 20))
            progress_bar.start()

            command = ["python", "./Final.py", "--input_file", input_file]

            def run_command():
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print("運行腳本時出錯：", e)
                else:
                    print("影像處理完成")

                    # 停止進度條動畫
                    progress_bar.stop()
                    progress_bar.destroy()
                    messagebox.showinfo("完成", "影像處理完成")

                    # 打開點雲查看器
                    show_point_cloud_viewer()
                finally:
                    process_button.config(state=tk.NORMAL)  # Re-enable the button after processing


            # Use the after method to ensure that progress bar updates take effect in the UI event loop
            root.after(10, run_command)

        print("Processing brain images")

    # Adjust the button magnification and make it in the middle of the UI
    process_button = tk.Button(root, text="選擇NIfTI檔案", command=process_and_run_final_pts, font=("Helvetica", 16, "bold"))
    process_button.pack(side=tk.TOP, pady=20)

    open_viewer_button = tk.Button(root, text="開啟點雲查看器", command=show_point_cloud_viewer, font=("Helvetica", 16, "bold"))
    open_viewer_button.pack(side=tk.TOP)

    # Centering the buttons
    root.update_idletasks()
    width = max(process_button.winfo_reqwidth(), open_viewer_button.winfo_reqwidth())
    process_button.pack_configure(anchor='n', pady=(0, (root.winfo_height() - title_label.winfo_height() - selected_file_label.winfo_height() - process_button.winfo_height() - open_viewer_button.winfo_height() - 20) // 2))
    open_viewer_button.pack_configure(anchor='n', pady=(0, (root.winfo_height() - title_label.winfo_height() - selected_file_label.winfo_height() - process_button.winfo_height() - open_viewer_button.winfo_height() - 20) // 2))

    root.mainloop()


def show_point_cloud_viewer():
    app = QApplication(sys.argv)
    viewer = PointCloudViewer()
    viewer.exec_()

if __name__ == '__main__':
    run_tkinter()


