##################################################
### Project of Group 1 : The Prediction of Job Change
### Date 05/05/2021
### Zongzhu Li
##################################################

#%%------------------------------------------------------
#import essencial packages

import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QPlainTextEdit,QDialog, QVBoxLayout, QSizePolicy, QMessageBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import random

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'

#%%-----------------------------------------------------------------------
# Deafault font size for all the windows
font_size_window = 'font-size:15px'

#%%------------------
# define RandomForest window
class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Random Forest Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9], self)

        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(True)

        self.labelPercentTest = QLabel('Percentage for Test :')
        self.labelPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText('30')

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0)
        self.groupBox1Layout.addWidget(self.feature7,3,1)
        self.groupBox1Layout.addWidget(self.feature8,4,0)
        self.groupBox1Layout.addWidget(self.feature9,4,1)
        self.groupBox1Layout.addWidget(self.labelPercentTest,5,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,5,1)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)

        self.groupBox2 = QGroupBox('Results for RF Model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.labelResults = QLabel('Results:')
        self.labelResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.labelAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.labelResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.labelAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes1=[self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        #::---------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------
        # Graphic 3 :Features Importance
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Features Importance')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        # #::--------------------------------------------
        # # Graphic 4 : ROC Curve by class
        # #::--------------------------------------------
        #
        # self.fig4 = Figure()
        # self.ax4 = self.fig4.add_subplot(111)
        # self.axes4 = [self.ax4]
        # self.canvas4 = FigureCanvas(self.fig4)
        #
        # self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #
        # self.canvas4.updateGeometry()
        #
        # self.groupBoxG4 = QGroupBox('ROC Curve by Class')
        # self.groupBoxG4Layout = QVBoxLayout()
        # self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        # self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBox2,0,1)
        self.layout.addWidget(self.groupBoxG1,1,1)
        self.layout.addWidget(self.groupBoxG2,0,2)
        self.layout.addWidget(self.groupBoxG3,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1000, 700)
        self.show()

    def update(self):

    # Random Forest Classifier
    # The parameters are processed to execute in the skit-learn Random Forest algorithm,
    # then the results are presented in graphics and reports in the canvas

        self.list_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = ff_happiness[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = ff_happiness[features_list[7]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[7]]],axis=1)


        vtest_per = float(self.txtPercentTest.text())/100

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        #vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt =  self.list_corr_features
        y_dt = ff_happiness["Happiness.Scale"]

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)

        # perform training with entropy.
        # Decision tree with entropy

        #specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=100)

        # perform training
        self.clf_rf.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::------------------------------------
        ##  Graph1 :
        ##  Confusion Matrix
        #::------------------------------------
        class_names1 = ['','Happy', 'Med.Happy', 'Low.Happy', 'Not.Happy']

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_rf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        n_classes = y_test_bin.shape[1]

        #From the sckict learn site
        #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2
        self.ax2.plot(fpr[2], tpr[2], color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Random Forest')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()
        ######################################
        # Graph - 3 Feature Importances
        #####################################
        # get feature importances
        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.list_corr_features.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        # #::-----------------------------------------------------
        # # Graph 4 - ROC Curve by Class
        # #::-----------------------------------------------------
        # str_classes= ['HP','MEH','LOH','NH']
        # colors = cycle(['magenta', 'darkorange', 'green', 'blue'])
        # for i, color in zip(range(n_classes), colors):
        #     self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
        #              label='{0} (area = {1:0.2f})'
        #                    ''.format(str_classes[i], roc_auc[i]))
        #
        # self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        # self.ax4.set_xlim([0.0, 1.0])
        # self.ax4.set_ylim([0.0, 1.05])
        # self.ax4.set_xlabel('False Positive Rate')
        # self.ax4.set_ylabel('True Positive Rate')
        # self.ax4.set_title('ROC Curve by Class')
        # self.ax4.legend(loc="lower right")
        #
        # # show the plot
        # self.fig4.tight_layout()
        # self.fig4.canvas.draw_idle()
        #
        # #::-----------------------------
        # # End of graph 4  - ROC curve by class
        # #::-----------------------------

# define KNN model
# class KNN(QMainWindow):

# define Logistic model
# class Logistic(QMainWindow):

# define HGB model
# class HGB(QMainWindow):

# define EDA1Graphs
# class EDA1Graphs(QMainWindow):

# define EDA2Graphs
# class EDA2Graphs(QMainWindow):

#::-------------------------------------------------------------
#:: Definition of a Class for the main manu in the application
#::-------------------------------------------------------------
class Menu(QMainWindow):

    def __init__(self):

        super().__init__()
        #::-----------------------
        #:: variables use to set the size of the window that contains the menu
        #::-----------------------
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300

        #:: Title for the application

        self.Title = 'Prediction of Job Change'

        #:: The initUi is call to create all the necessary elements for the menu

        self.initUI()

    def initUI(self):

        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        #::-----------------------------
        # 1. Create the menu bar
        # 2. Create an item in the menu bar
        # 3. Creaate an action to be executed the option in the  menu bar is choosen
        #::-----------------------------
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        #:: Add another option to the Menu Bar

        EDAsWin = mainMenu.addMenu ('EDAs')

        #:: Add another option to the Menu Bar

        ModelsWin = mainMenu.addMenu('Models')

        #::--------------------------------------
        # Exit action
        # The following code creates the the da Exit Action along
        # with all the characteristics associated with the action
        # The Icon, a shortcut , the status tip that would appear in the window
        # and the action
        #  triggered.connect will indicate what is to be done when the item in
        # the menu is selected
        # These definitions are not available until the button is assigned
        # to the menu
        #::--------------------------------------

        exitButton = QAction('&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        #:: This line adds the button (item element ) to the menu

        fileMenu.addAction(exitButton)

        #::----------------------------------------------------
        #::Add Example 1 We create the item Menu Example1
        #::This option will present a message box upon request
        #::----------------------------------------------------

        EDA1Button = QAction("Overall Percentage of Looking for Job Change  ", self)
        EDA1Button.setStatusTip("Show Statistic of Looking for Job Change")
        EDA1Button.triggered.connect(self.EDA1)
        EDAsWin.addAction(EDA1Button) # We addd the EDA1Button action to the Menu EDAs

        EDA2Button = QAction("Features vs Target ", self)
        #EDA2Button.setStatusTip("Show EDA2")
        EDA2Button.triggered.connect(self.EDA2)
        EDAsWin.addAction(EDA2Button) # We addd the EDA2Button action to the Menu EDAs

        LogisticButton = QAction("Logistic", self)
        LogisticButton.setStatusTip("Show Logistic Results")
        LogisticButton.triggered.connect(self.Logistic)
        ModelsWin.addAction(LogisticButton) # We add the LogisticButton to the Menu models

        RandomForestButton = QAction("RandomForest", self)
        RandomForestButton.setStatusTip("Show RandomForest Results")
        RandomForestButton.triggered.connect(self.RandomForest)
        ModelsWin.addAction(RandomForestButton) # We add the RandomForestButton to the Menu models

        HGBButton = QAction("HGB", self)
        HGBButton.setStatusTip("Show HGB Results")
        HGBButton.triggered.connect(self.HGB)
        ModelsWin.addAction(HGBButton) # We add the HGB Button to the Menu models

        KNNButton = QAction("KNN", self)
        KNNButton.setStatusTip("Show KNN Results")
        KNNButton.triggered.connect(self.KNN)
        ModelsWin.addAction(KNNButton) # We add the KNNButton to the Menu models
        self.show()

    def EDA1(self):
        QMessageBox.about(self, "Results Example1", "Hello World!!!")
            #::---------------------------------------------------------
        # This class creates a graph using the target in the dataset
        #::---------------------------------------------------------
        # dialog = EDA1Graphs()
        # self.dialogs.append(dialog)
        # dialog.show()

    def EDA2(self):
        #::---------------------------------------------------------
        # This class creates graphs using the target and the features in the dataset
        #::---------------------------------------------------------
        dialog = EDA2Graphs()
        self.dialogs.append(dialog)
        dialog.show()

    def RandomForest(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def Logistic(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Logistic Classifier Algorithm
        #::-------------------------------------------------------------
        dialog = Logistic()
        self.dialogs.append(dialog)
        dialog.show()

    def HGB(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the HGB Classifier Algorithm
        #::-------------------------------------------------------------
        dialog = HGB()
        self.dialogs.append(dialog)
        dialog.show()

    def KNN(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the KNN Classifier Algorithm
        #::-------------------------------------------------------------
        dialog = KNN()
        self.dialogs.append(dialog)
        dialog.show()
#::------------------------
#:: Application starts here
#::------------------------
def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = Menu()  # Cretes the menu
    sys.exit(app.exec_())  # Close the application

if __name__ == '__main__':
    main()