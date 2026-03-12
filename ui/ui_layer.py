from __future__ import annotations

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


STYLE = """
QWidget {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
QWidget#root { background-color: #F5F5F5; }

QWidget#sidebar {
    background-color: #2C2C2C;
    min-width: 220px; max-width: 220px;
}
QLabel#appTitle { color:#FFFFFF; font-size:18px; font-weight:bold; padding:20px 16px 8px 16px; }
QLabel#appVersion { color:#F5A623; font-size:11px; padding:0px 16px 20px 16px; }
QLabel#menuItem { color:#CCCCCC; font-size:14px; padding:12px 20px; }
QLabel#menuItemActive {
    color:#FFFFFF; font-size:14px; font-weight:bold; padding:12px 20px;
    background-color:#3A3A3A; border-left:3px solid #F5A623;
}
QWidget#topbar { background-color:#FFFFFF; border-bottom:1px solid #E0E0E0; }
QWidget#content { background-color:#F5F5F5; }
QLabel#pageTitle { color:#1A1A1A; font-size:24px; font-weight:bold; }
QWidget#card { background-color:#FFFFFF; border-radius:8px; }

QPushButton#backBtn {
    background-color:#FFFFFF; color:#1A1A1A;
    border:1px solid #DDDDDD; border-radius:6px; font-size:13px; padding:8px 18px;
}
QPushButton#backBtn:hover { background-color:#F5F5F5; }

QLabel#imageArea {
    background-color:#F0F0F0;
    border-radius:6px;
    border:2px solid #E8E8E8;
}
"""


class LayerScreen(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setObjectName("root")
        self.setStyleSheet(STYLE)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(0)

        title = QLabel("짐 싸기")
        title.setObjectName("appTitle")
        version = QLabel("v1.1.0")
        version.setObjectName("appVersion")
        sb.addWidget(title)
        sb.addWidget(version)

        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet("background:#3D3D3D;")
        sb.addWidget(div)

        for icon, name, active in [
            ("🏠", "홈", False),
            ("📷", "사진 입력", False),
            ("✂️", "세그멘테이션", False),
            ("🗂", "레이어 뷰", True),
        ]:
            lbl = QLabel(f"  {icon}  {name}")
            lbl.setObjectName("menuItemActive" if active else "menuItem")
            sb.addWidget(lbl)

        sb.addStretch()
        settings_lbl = QLabel("  ⚙️  설정")
        settings_lbl.setObjectName("menuItem")
        sb.addWidget(settings_lbl)

        content = QWidget()
        content.setObjectName("content")
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(0)

        topbar = QWidget()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(56)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(24, 0, 24, 0)
        tb.addStretch()
        admin = QLabel("admin  ▾")
        admin.setStyleSheet("color:#1A1A1A;font-size:14px;")
        tb.addWidget(admin)

        body = QWidget()
        b_layout = QVBoxLayout(body)
        b_layout.setContentsMargins(32, 32, 32, 32)
        b_layout.setSpacing(20)

        header = QHBoxLayout()
        page_title = QLabel("레이어 뷰")
        page_title.setObjectName("pageTitle")
        self.back_button = QPushButton("← 뒤로가기")
        self.back_button.setObjectName("backBtn")
        self.back_button.clicked.connect(self.go_back)
        header.addWidget(page_title)
        header.addStretch()
        header.addWidget(self.back_button)

        card = QWidget()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        self.image_label = QLabel()
        self.image_label.setObjectName("imageArea")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(420)
        self.image_label.setText("레이어 이미지가 여기에 표시됩니다.")
        self.image_label.setStyleSheet(
            "QLabel#imageArea{background:#F5F5F5;border-radius:6px;"
            "border:2px dashed #DDDDDD;color:#AAAAAA;font-size:14px;}"
        )

        legend = QLabel("현재 화면은 전처리된 이미지를 바탕으로 남아 있는 mask를 오버레이한 결과입니다.")
        legend.setStyleSheet("font-size:12px;color:#666666;padding:2px 0;")
        legend.setWordWrap(True)

        card_layout.addWidget(self.image_label)
        card_layout.addWidget(legend)

        b_layout.addLayout(header)
        b_layout.addWidget(card)
        b_layout.addStretch()

        c_layout.addWidget(topbar)
        c_layout.addWidget(body)

        root.addWidget(sidebar)
        root.addWidget(content)

    def go_back(self):
        self.stack.setCurrentIndex(2)

    def set_data(self, base_image, outputs):
        if base_image is None:
            self.image_label.setText("표시할 이미지가 없습니다.")
            self.image_label.setPixmap(QPixmap())
            return

        overlay = base_image.copy()
        for obj in outputs:
            mask = obj.get("mask")
            if mask is None:
                continue
            overlay[mask] = [255, 80, 0]

        blended = cv2.addWeighted(base_image, 0.65, overlay, 0.35, 0)
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pixmap.scaled(720, 460, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
