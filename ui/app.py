from PyQt5.QtWidgets import QMainWindow
from ui.design import Ui_IANN


class APP_IANN(QMainWindow, Ui_IANN):
    def __init__(self, parent=None):
        super(APP_IANN, self).__init__(parent)
        self.setupUi(self)
        ## 信号
        self.btnOpenImage.clicked.connect(self.check_click)  # 打开图像
        self.btnOpenFolder.clicked.connect(self.check_click)  # 打开文件夹
        self.btnUndo.clicked.connect(self.check_click)  # 撤销
        self.btnRedo.clicked.connect(self.check_click)  # 重做
        self.btnUndoAll.clicked.connect(self.check_click)  # 撤销全部
        self.btnAbout.clicked.connect(self.check_click)  # 关于
        # 细粒度（这种可以通过sender的text来知道哪个键被点击了）
        for action in self.btnScale.Menu.actions():
            action.triggered.connect(self.check_click)
        # 帮助
        for action in self.btnHelp.Menu.actions():
            action.triggered.connect(self.check_click)
        self.btnSLeft.clicked.connect(self.check_click)  # 上一张图
        self.btnSRight.clicked.connect(self.check_click)  # 下一张图
        # 选择模型
        for action in self.btnModelSelect.Menu.actions():
            action.triggered.connect(self.update_model_name)
        self.listLabel.clicked.connect(self.check_click)  # 数据列表选择（用row可以获取点击的行数）
        self.listClass.clicked.connect(self.check_click)  # 标签选择
        self.btnAddClass.clicked.connect(self.check_click)  # 添加标签
        # 分别滑动三个滑动滑块
        self.sldMask.sliderReleased.connect(self.slider_2_label)
        self.sldSeg.sliderReleased.connect(self.slider_2_label)
        self.sldPointSzie.sliderReleased.connect(self.slider_2_label)
        self.btnSave.clicked.connect(self.check_click)  # 保存
        
    # 确认点击
    def check_click(self):
        print(self.sender().text())

    # 滑块数值与标签数值同步
    def slider_2_label(self):
        slider = self.sender()
        name = slider.objectName()
        if name == "sldMask":
            self.labMaskShow.setText(str(slider.value() / 10.))
        elif name == "sldSeg":
            self.labSegShow.setText(str(slider.value() / 10.))
        else:
            self.labPointSizeShow.setText(str(slider.value()))

    # 当前打开的模型名称或类别更新
    def update_model_name(self):
        self.labModelName.setText(self.sender().text())
        self.check_click()