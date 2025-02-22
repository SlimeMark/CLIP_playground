import sys
import torch
import clip
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit, QListWidget, QListWidgetItem, QHBoxLayout, QProgressBar, QSpinBox, QLineEdit
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
import pickle

download_root = "./models"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root=download_root)

CACHE_FILE = "image_cache.pkl"

class ImageProcessor(QThread):
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(dict, list)

    def __init__(self, image_files, query_text, batch_size, cache):
        super().__init__()
        self.image_files = image_files
        self.query_text = query_text
        self.batch_size = batch_size
        self.cache = cache

    def run(self):
        text = clip.tokenize([self.query_text]).to(device)
        images = []
        images_files = []

        self.progress.emit(0)
        for i, img_path in enumerate(self.image_files):
            if img_path in self.cache:
                image = self.cache[img_path]
            else:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                self.cache[img_path] = image
            images_files.append(img_path)
            images.append(image)
            self.progress.emit(int((i + 1) / len(self.image_files) * 100))

        results = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch_images = torch.cat(images[i:i+self.batch_size])
                image_features = model.encode_image(batch_images)
                text_features = model.encode_text(text)
                
                batch_results = [torch.cosine_similarity(img_feat, text_features).item() for img_feat in image_features]
                results.extend(batch_results)
                self.progress.emit(int((i + self.batch_size) / len(images) * 100))

        # 将余弦相似度转换为匹配度百分比
        match_scores = {img_path: score * 100 for img_path, score in zip(images_files, results)}

        self.result_ready.emit(match_scores, self.image_files)

class CLIPImageSearch(QWidget):
    def __init__(self):
        super().__init__()
        self.cache = self.load_cache()
        self.image_files = []
        self.results = {}
        self.low_score_results = {}
        self.showing_low_scores = False  # 新增变量
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Enter text description...")
        self.text_input.setFixedHeight(100)  # 设置文本框高度
        left_layout.addWidget(self.text_input)

        self.folder_path = QLineEdit(self)
        self.folder_path.setPlaceholderText("Select image folder...")
        left_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.select_folder)
        left_layout.addWidget(self.browse_button)

        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setRange(1, 64)
        self.batch_size_input.setValue(16)
        left_layout.addWidget(QLabel("Batch Size:"))
        left_layout.addWidget(self.batch_size_input)

        self.search_button = QPushButton("Search", self)
        self.search_button.clicked.connect(self.run_clip_search)
        left_layout.addWidget(self.search_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.progress_bar)

        self.load_low_score_button = QPushButton("Load low score matches", self)
        self.load_low_score_button.setEnabled(False)
        self.load_low_score_button.clicked.connect(self.display_low_score_results)
        left_layout.addWidget(self.load_low_score_button)

        main_layout.addLayout(left_layout, 1)

        right_layout = QVBoxLayout()
        self.result_list = QListWidget(self)
        self.result_list.itemDoubleClicked.connect(self.open_image)
        right_layout.addWidget(self.result_list)
        main_layout.addLayout(right_layout, 2)

        self.setLayout(main_layout)
        self.setWindowTitle("CLIP Image Search")
        self.resize(800, 600)

        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            }
            QLineEdit, QTextEdit, QListWidget, QProgressBar, QSpinBox {
            background-color: #2d2d2d;
            border: 1px solid #3d3d3d;
            border-radius: 5px;
            padding: 5px;
            color: #ffffff;
            }
            QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 5px;
            padding: 10px;
            color: #ffffff;
            font-weight: bold;
            }
            QPushButton:hover {
            background-color: #4a4a4a;
            }
            QLabel {
            font-weight: bold;
            }
            QProgressBar {
            text-align: center;
            background-color: #2d2d2d;
            border-radius: 5px;
            }
            QProgressBar::chunk {
            background-color: #0078d7;
            border-radius: 5px;
            }
        """)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_path.setText(folder)

    def run_clip_search(self):
        query_text = self.text_input.toPlainText()
        image_folder = self.folder_path.text()
        batch_size = self.batch_size_input.value()
        
        if not query_text or not image_folder:
            return
        
        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith((".jpg", ".jpeg", ".png"))]
        self.results = {}
        self.low_score_results = {}
        
        self.progress_bar.setValue(0)
        self.image_processor = ImageProcessor(self.image_files, query_text, batch_size, self.cache)
        self.image_processor.progress.connect(self.progress_bar.setValue)
        self.image_processor.result_ready.connect(self.update_results)
        self.image_processor.start()

    def update_results(self, result_dict, processed_files):
        self.results.update(result_dict)
        self.display_results()

    def display_results(self):
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        self.result_list.clear()
        threshold = np.percentile(list(self.results.values()), 75)
        for img_path, score in sorted_results:
            if score >= threshold:
                self.add_image_result(img_path, score)
            else:
                self.low_score_results[img_path] = score
        self.load_low_score_button.setEnabled(bool(self.low_score_results))
        self.load_low_score_button.setText("Load low score matches")  # 重置按钮文本
        self.showing_low_scores = False  # 重置状态
        self.save_cache()

    def display_low_score_results(self):
        if self.showing_low_scores:
            self.display_results()
            self.load_low_score_button.setText("Load low score matches")
        else:
            sorted_results = sorted(self.low_score_results.items(), key=lambda x: x[1], reverse=True)
            self.result_list.clear()
            for img_path, score in sorted_results:
                self.add_image_result(img_path, score)
            self.load_low_score_button.setText("Show high score results only")
        self.showing_low_scores = not self.showing_low_scores  # 切换状态

    def add_image_result(self, img_path, score):
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, img_path)
        widget = QWidget()
        layout = QHBoxLayout()

        img_label = QLabel()
        pixmap = QPixmap(img_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
        img_label.setPixmap(pixmap)

        # 使用对数函数对匹配度百分比进行加权
        if score < 23:
            weighted_score = "Low"
        else:
            weighted_score = f"{100 * (1 - np.exp(-score / 25)):.2f}%"
        
        text_label = QLabel(f"{os.path.basename(img_path)}\nScore: {weighted_score} ({score:.2f}%)")
        
        layout.addWidget(img_label)
        layout.addWidget(text_label)
        layout.addStretch()
        widget.setLayout(layout)

        item.setSizeHint(widget.sizeHint())
        self.result_list.addItem(item)
        self.result_list.setItemWidget(item, widget)

    def open_image(self, item):
        img_path = item.data(Qt.ItemDataRole.UserRole)
        QDesktopServices.openUrl(QUrl.fromLocalFile(img_path))

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(self.cache, f)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CLIPImageSearch()
    window.show()
    sys.exit(app.exec())