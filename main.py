# main.py
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QFileDialog,
    QTabWidget, QFormLayout, QLineEdit
)
from PySide6.QtCore import Qt
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4


class RiskCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ризик сталевар v1.0")
        self.resize(1200, 800)

        # Дані
        self.data = None

        # Інтерфейс
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.create_input_tab()
        self.create_results_tab()
        self.create_plots_tab()
        self.create_report_tab()

    def create_input_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Таблиця вводу
        self.table = QTableWidget(12, 14)
        self.table.setHorizontalHeaderLabels([
            'Fi', 'Розмір, нм', 'C', 'CR', 'EF', 'ED', 'BW', 'AT', '365',
            'LADD', 'Ri', 'ELNCRi', 'HQ', 'Кт', 'Кр'
        ])
        layout.addWidget(self.table)

        # Поля для вводу SF / RfD
        form = QFormLayout()
        self.sf_nitro = QLineEdit("49")
        self.sf_benzene = QLineEdit("0.027")
        self.rfd = QLineEdit("0.0001")  # Приклад RfD
        form.addRow("SF N-нітрозометиламін:", self.sf_nitro)
        form.addRow("SF бензолу:", self.sf_benzene)
        form.addRow("RfD (для HQ):", self.rfd)
        layout.addLayout(form)

        # Кнопки
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Завантажити Excel")
        btn_calc = QPushButton("Розрахувати")
        btn_save = QPushButton("Зберегти дані")

        btn_load.clicked.connect(self.load_excel)
        btn_calc.clicked.connect(self.calculate)
        btn_save.clicked.connect(self.save_results)

        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_calc)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

        widget.setLayout(layout)
        self.tabs.addTab(widget, "Введення даних")

    def load_excel(self):
        file, _ = QFileDialog.getOpenFileName(self, "Відкрити Excel", "", "Excel Files (*.xlsx)")
        if not file: return
        df = pd.read_excel(file)
        self.fill_table_from_df(df)

    def fill_table_from_df(self, df):
        df = df.fillna(0)
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                if col in ['Fi', 'C', 'CR', 'EF', 'ED', 'BW', 'AT', 'LADD', 'Ri', 'ELNCRi', 'HQ', 'Кт', 'Кр', 'Розмір, нм']:
                    val = row[col]
                    self.table.setItem(i, self.col_index(col), QTableWidgetItem(f"{val}"))

    def col_index(self, name):
        headers = ['Fi', 'Розмір, нм', 'C', 'CR', 'EF', 'ED', 'BW', 'AT', '365', 'LADD', 'Ri', 'ELNCRi', 'HQ', 'Кт', 'Кр']
        return headers.index(name)

    def calculate(self):
        df = self.get_table_data()
        sf_nitro = float(self.sf_nitro.text())
        sf_benzene = float(self.sf_benzene.text())
        rfd = float(self.rfd.text())

        # LADD (загальний)
        df['LADD'] = (df['C'] * df['CR'] * df['EF'] * df['ED']) / (df['BW'] * df['AT'] * 365)

        # Ri (токсична доза)
        df['Ri'] = df['LADD'] * df['Кт']

        # ELNCRi (неканцерогенний ризик)
        df['ELNCRi'] = df['Ri']

        # HQ
        df['HQ'] = df['Ri'] / rfd

        # CR (для бензолу та N-нітрозометиламіну)
        total_c = df['C'].sum()
        ladd_total = (total_c * df['CR'].mean() * df['EF'].mean() * df['ED'].mean()) / (70 * 70 * 365)
        cr_nitro = ladd_total * sf_nitro
        cr_benzene = ladd_total * sf_benzene

        # Загальний неканцерогенний ризик
        total_hq = df['HQ'].sum()
        total_risk = df['ELNCRi'].sum()

        self.data = {
            'df': df,
            'cr_nitro': cr_nitro,
            'cr_benzene': cr_benzene,
            'total_hq': total_hq,
            'total_risk': total_risk
        }

        self.update_results_tab()
        self.update_plots_tab()
        self.update_report_tab()

        # Оновити таблицю
        self.update_table(df)

    def update_table(self, df):
        for i, row in df.iterrows():
            for j, col in enumerate(['Fi', 'Розмір, нм', 'C', 'CR', 'EF', 'ED', 'BW', 'AT', '365', 'LADD', 'Ri', 'ELNCRi', 'HQ', 'Кт', 'Кр']):
                self.table.setItem(i, self.col_index(col), QTableWidgetItem(f"{row[col]:.2e}"))

    def get_table_data(self):
        rows = []
        for i in range(self.table.rowCount()):
            row = {}
            for j, col in enumerate(['Fi', 'Розмір, нм', 'C', 'CR', 'EF', 'ED', 'BW', 'AT', '365', 'LADD', 'Ri', 'ELNCRi', 'HQ', 'Кт', 'Кр']):
                item = self.table.item(i, j)
                try:
                    row[col] = float(item.text()) if item else 0
                except:
                    row[col] = 0
            rows.append(row)
        return pd.DataFrame(rows)

    def create_results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.results_label = QLabel("Результати з'являться після розрахунку")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-size: 14px; padding: 10px;")
        self.results_label.setTextInteractionFlags(Qt.TextBrowserInteraction)  # Дозволяє клацати по тексту
        layout.addWidget(self.results_label)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Результати")

    def update_results_tab(self):
        d = self.data
        text = f"""
        <h2>Результати розрахунку</h2>
        <b>Канцерогенний ризик:</b><br>
        &bull; N-нітрозометиламін: <b>{d['cr_nitro']:.2e}</b><br>
        &bull; Бензол: <b>{d['cr_benzene']:.2e}</b><br><br>
        <b>Неканцерогенний ризик:</b><br>
        &bull; Загальний HQ: <b>{d['total_hq']:.3f}</b><br>
        &bull; Загальний ELNCR: <b>{d['total_risk']:.3f}</b><br><br>
        <b>Висновок:</b><br>
        """
        if d['cr_nitro'] < 1e-4:
            text += "Ризик від N-нітрозометиламіну — <span style='color:green'>прийнятний</span><br>"
        else:
            text += "Ризик від N-нітрозометиламіну — <span style='color:red'>неприйнятний</span><br>"

        if d['cr_benzene'] < 1e-4:
            text += "Ризик від бензолу — <span style='color:green'>прийнятний</span><br>"
        else:
            text += "Ризик від бензолу — <span style='color:red'>неприйнятний</span><br>"

        if d['total_hq'] > 1:
            text += "Неканцерогенний ризик (HQ) перевищує норму — <span style='color:red'>потрібні заходи</span>"
        else:
            text += "Неканцерогенний ризик у межах норми — <span style='color:green'>безпечний</span>"

        self.results_label.setText(text)

    def create_plots_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.plot_btn = QPushButton("Показати графіки")
        self.plot_btn.clicked.connect(self.plot_results)
        layout.addWidget(self.plot_btn)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Графіки")

    def plot_results(self):
        if self.data is None: return
        df = self.data['df']
        sns.set(style="whitegrid")

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # HQ vs Розмір
        ax[0,0].bar(df['Розмір, нм'], df['HQ'], color='skyblue')
        ax[0,0].set_title('Hazard Quotient (HQ) за розміром частинок')
        ax[0,0].set_xlabel('Розмір, нм')
        ax[0,0].set_ylabel('HQ')

        # LADD
        ax[0,1].plot(df['Розмір, нм'], df['LADD'], marker='o', color='green')
        ax[0,1].set_title('LADD за розміром')
        ax[0,1].set_xlabel('Розмір, нм')
        ax[0,1].set_ylabel('LADD')

        # Діаграма внеску
        positive_hq = df[df['HQ'] > 0]
        ax[1,0].pie(positive_hq['HQ'], labels=positive_hq['Розмір, нм'].astype(int), autopct='%1.1f%%')
        ax[1,0].set_title('Внесок фракцій у HQ')

        # CR порівняння
        cr_values = [self.data['cr_nitro'], self.data['cr_benzene']]
        substances = ['N-нітрозометиламін', 'Бензол']
        ax[1,1].bar(substances, cr_values, color=['red', 'orange'])
        ax[1,1].set_title('Канцерогенний ризик')
        ax[1,1].set_ylabel('CR')
        ax[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def create_report_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.report_btn = QPushButton("Згенерувати PDF-звіт")
        self.report_btn.clicked.connect(self.generate_report)
        layout.addWidget(self.report_btn)
        widget.setLayout(layout)
        self.tabs.addTab(widget, "Звіт")

    def generate_report(self):
        if self.data is None: return
        filename, _ = QFileDialog.getSaveFileName(self, "Зберегти PDF", "звіт_ризик.pdf", "PDF Files (*.pdf)")
        if not filename: return

        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        flowables = []

        flowables.append(Paragraph("Звіт з оцінки професійного ризику", styles['Title']))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("1. Висновки", styles['Heading2']))
        flowables.append(Paragraph(f"Канцерогенний ризик (N-нітрозометиламін): {self.data['cr_nitro']:.2e}", styles['Normal']))
        flowables.append(Paragraph(f"Канцерогенний ризик (бензол): {self.data['cr_benzene']:.2e}", styles['Normal']))
        flowables.append(Paragraph(f"Загальний HQ: {self.data['total_hq']:.3f}", styles['Normal']))
        flowables.append(Spacer(1, 12))

        flowables.append(Paragraph("2. Дані", styles['Heading2']))
        df = self.data['df']
        data = [list(df.columns)] + df.round(6).values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        flowables.append(table)

        doc.build(flowables)
        self.statusBar().showMessage(f"Звіт збережено: {filename}", 3000)

    def save_results(self):
        if self.data is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Зберегти Excel", "результати.xlsx", "Excel Files (*.xlsx)")
        if filename:
            self.data['df'].to_excel(filename, index=False)
            self.statusBar().showMessage(f"Дані збережено: {filename}", 3000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RiskCalculator()
    window.show()
    sys.exit(app.exec_())