#region IMPORTS
import tkinter as tk
from tkinter import filedialog, messagebox,ttk,Menu
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier
from sklearn.cluster import AgglomerativeClustering,KMeans
from catboost import CatBoostClassifier
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score,confusion_matrix,roc_curve, roc_auc_score, precision_recall_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import sys
import io
import numpy as np
from collections import Counter
import joblib
#endregion

class MLMasterApp:

    #region CONSTRUCTION
    def __init__(self, root):
        self.root = root
        self.root.title("ML MASTER")
        self.root.geometry("800x600")
        self.canvas = None
        self.selected_page = tk.StringVar(value="Data Preprocessing")
        self.file_loaded = False  

        self.create_widgets()
        self.hide_preprocessing_widgets()
        self.update_navigation_buttons()

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.rowconfigure(4, weight=1)
        self.root.rowconfigure(5, weight=5)
    #endregion

    #region GUI
    def create_widgets(self):
        self.title_label = tk.Label(self.root, text="ML MASTER", font=("Arial", 16))
        self.title_label.grid(row=0, column=0, padx=10, pady=10, sticky='nw')

        self.right_buttons_frame = tk.Frame(self.root)
        self.right_buttons_frame.grid(row=0, column=1, padx=10, pady=10, sticky='ne')

        self.data_preprocessing_button = tk.Button(self.right_buttons_frame, text="Data Preprocessing", command=self.go_to_data_preprocessing, cursor="hand2")
        self.data_preprocessing_button.grid(row=0, column=0, padx=5, pady=10, sticky='ne')
        
        self.feature_engineering_button = tk.Button(self.right_buttons_frame, text="Feature Engineering", command=self.go_to_feature_engineering, cursor="hand2")
        self.feature_engineering_button.grid(row=0, column=1, padx=5, pady=10, sticky='ne')

        self.data_viz_button = tk.Button(self.right_buttons_frame, text="Data Visualization", command=self.go_to_data_viz, cursor="hand2")
        self.data_viz_button.grid(row=0, column=2, padx=5, pady=10, sticky='ne')

        self.model_training_button = tk.Button(self.right_buttons_frame, text="Model Training", command=self.go_to_model_training, cursor="hand2")
        self.model_training_button.grid(row=0, column=3, padx=5, pady=10, sticky='ne')

        self.load_file_button = tk.Button(self.root, text="Dosya Yükle", command=self.load_file, cursor="hand2")
        self.load_file_button.grid(row=1, column=0, padx=10, pady=5, sticky='w')

        self.analysis_frame = tk.Frame(self.root)

        self.analysis_options = ["İlk 5 Satır", "Columnlar", "Shape","Info", "Veri Tipleri", "Null Değerler","Unique","N-Unique","Korelasyon Matrisi","İstatistiksel Bilgiler","Aykırı Verileri Ayıkla"]
        self.analysis_var = tk.StringVar(value=self.analysis_options[0])
        self.analysis_menu = ttk.Combobox(self.analysis_frame, textvariable=self.analysis_var, values=self.analysis_options, cursor="hand2")
        self.analysis_menu.pack(side='left')

        self.analysis_button = tk.Button(self.analysis_frame, text="Göster", command=self.analyze_data, cursor="hand2")
        self.analysis_button.pack(side='left', padx=5)

        self.fill_na_button = tk.Button(self.root, text="Boş Verileri Doldur", command=self.fill_na, cursor="hand2")
        self.save_dataset_button = tk.Button(self.root, text="Güncel Veriyi Kaydet", command=self.save_dataset, cursor="hand2")


        self.query_input_grid = tk.Frame(self.root)
        self.query_input_grid.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky='w')

        tk.Label(self.query_input_grid, text="Sıralama:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        tk.Label(self.query_input_grid, text="Gruplama:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        tk.Label(self.query_input_grid, text="Filtreleme:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        tk.Label(self.query_input_grid,text="Filtre Değeri:").grid(row=3,column=0,padx=5,pady=5,sticky="w")
        tk.Label(self.query_input_grid,text="Filtre Koşulu:").grid(row=4,column=0,padx=5,pady=5,sticky="w")

        self.sort_combobox = ttk.Combobox(self.query_input_grid, values=[], state="readonly")
        self.sort_combobox.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.group_combobox = ttk.Combobox(self.query_input_grid, values=[], state="readonly")
        self.group_combobox.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.filter_combobox = ttk.Combobox(self.query_input_grid, values=[], state="readonly")
        self.filter_combobox.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        self.filter_value_input = tk.Entry(self.query_input_grid)
        self.filter_value_input.grid(row=3, column=1, padx=5, pady=5, sticky='w')
         
        self.filter_condition_combobox = ttk.Combobox(self.query_input_grid, values=["<",">","=","<=",">=","!="], state="readonly")
        self.filter_condition_combobox.grid(row=4, column=1, padx=5, pady=5, sticky='w')

        self.query_button = tk.Button(self.query_input_grid, text="Sorgula", command=self.execute_query, cursor="hand2")
        self.query_button.grid(row=0, column=2, rowspan=3, padx=10, pady=5, sticky='w')

        self.result_frame = tk.Frame(self.root)
        self.result_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10)
                   
        self.tree = ttk.Treeview(self.result_frame)
        self.tree.grid(row=0, column=0, sticky='nsew')

        self.vsb = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.tree.yview)
        self.vsb.grid(row=0, column=1, sticky='ns')
        self.hsb = ttk.Scrollbar(self.result_frame, orient="horizontal", command=self.tree.xview)
        self.hsb.grid(row=1, column=0, sticky='ew')

        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        self.result_frame.grid_rowconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>", self.on_double_click_unique)
    
    def hide_preprocessing_widgets(self):
        self.analysis_frame.grid_remove()
        self.fill_na_button.grid_remove()
        self.save_dataset_button.grid_remove()
        self.result_frame.grid_remove()
        self.query_input_grid.grid_remove()

    def show_preprocessing_widgets(self):
        self.analysis_frame.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        self.fill_na_button.grid(row=4, column=0, padx=10, pady=5, sticky='w')
        self.save_dataset_button.grid(row=5, column=0, padx=10, pady=5, sticky='w')
        self.query_input_grid.grid(row=4, column=1, padx=10, pady=5, sticky='w')
        self.result_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

    def hide_load_file_button(self):
        self.load_file_button.grid_remove()
    
    def hide_plot_details(self):
        if hasattr(self, 'plot_details_frame'):
            self.plot_details_frame.grid_remove()
     
    def hide_feature_engineering(self):
        if hasattr(self,"feature_engineering_frame"):
            self.feature_engineering_frame.grid_remove()

    def hide_model_details(self):
        if hasattr(self, 'model_details_frame'):
            self.model_details_frame.grid_remove()
            
    def show_load_file_button(self):
        self.load_file_button.grid()
    
    def update_comboboxes(self):
        columns = list(self.data.columns)
        self.sort_combobox['values'] = columns
        self.group_combobox['values'] = columns
        self.filter_combobox['values'] = columns
    
    def go_to_data_preprocessing(self):
        self.selected_page.set("Data Preprocessing")
        self.update_navigation_buttons()
        self.show_load_file_button()
        if self.file_loaded:
            self.show_preprocessing_widgets()
        else:
            self.hide_preprocessing_widgets()
        self.hide_plot_details()
        self.hide_model_details()
        self.hide_feature_engineering()
        if self.canvas:
            self.canvas.get_tk_widget().grid_remove()
    
    def go_to_feature_engineering(self):
        self.selected_page.set("Feature Engineering")
        self.update_navigation_buttons()
        self.hide_preprocessing_widgets()
        self.hide_load_file_button()
        self.hide_plot_details()
        self.hide_model_details()
        if self.canvas: 
            self.canvas.get_tk_widget().grid_remove()
        self.hide_plot_details()
        if self.file_loaded:
            self.feature_engineering_page() 
        else:
            messagebox.showinfo("Dosya Yükle", "Henüz bir dosya yüklemediniz. Lütfen Data Preprocessing sayfasından dosya yükleyiniz.")
    
    def go_to_model_training(self):
        self.selected_page.set("Model Training")
        self.update_navigation_buttons()
        self.hide_preprocessing_widgets()
        self.hide_load_file_button()
        self.hide_model_details()
        self.hide_feature_engineering()
        if self.canvas: 
            self.canvas.get_tk_widget().grid_remove()
        self.hide_plot_details()
        if self.file_loaded:
            self.ask_model_details() 
        else:
            messagebox.showinfo("Dosya Yükle", "Henüz bir dosya yüklemediniz. Lütfen Data Preprocessing sayfasından dosya yükleyiniz.")

    def go_to_data_viz(self):
        self.selected_page.set("Data Visualization")
        self.update_navigation_buttons()
        self.hide_preprocessing_widgets()
        self.hide_model_details()
        self.hide_load_file_button()
        self.hide_feature_engineering()
        if self.file_loaded:
            self.ask_plot_details()
            if self.canvas:
                self.canvas.get_tk_widget().grid(row=2, column=1, padx=10, pady=10)
        else:
            messagebox.showinfo("Dosya Yükle", "Henüz bir dosya yüklemediniz. Lütfen Data Preprocessing sayfasından dosya yükleyiniz.")

    def update_navigation_buttons(self):
        self.data_preprocessing_button.config(relief=tk.SUNKEN if self.selected_page.get() == "Data Preprocessing" else tk.RAISED)
        self.data_viz_button.config(relief=tk.SUNKEN if self.selected_page.get() == "Data Visualization" else tk.RAISED)
        self.model_training_button.config(relief=tk.SUNKEN if self.selected_page.get() == "Model Training" else tk.RAISED)
        self.feature_engineering_button.config(relief=tk.SUNKEN if self.selected_page.get() == "Feature Engineering" else tk.RAISED)

    def clear_tree(self):
        self.tree.delete(*self.tree.get_children())
    #endregion

    #region DATA PREPROCESSING
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("Excel files", "*.xlsx")])
        if file_path:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Başarılı", "CSV dosyası yüklendi.")
                self.show_raw_data()
                self.update_comboboxes()
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
                messagebox.showinfo("Başarılı", "JSON dosyası yüklendi.")
                self.show_raw_data()
                self.update_comboboxes()
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
                messagebox.showinfo("Başarılı", "XLSX dosyası yüklendi.")
                self.show_raw_data()
                self.update_comboboxes()
            else:
                messagebox.showerror("Hata", "Desteklenmeyen dosya formatı.")
                return
            self.file_loaded = True
            self.show_preprocessing_widgets()
    
    def analyze_data(self):
        self.clear_tree()
        option = self.analysis_var.get()
        if option == "İlk 5 Satır":
            self.show_raw_data()
        elif option == "Columnlar":
            self.show_columns()
        elif option == "Shape":
            self.show_in_text_widget(str(self.data.shape))
        elif option == "Veri Tipleri":
            self.show_data_types()
        elif option == "Null Değerler":
            self.show_null_values()
        elif option == "Korelasyon Matrisi":
            self.show_corr_matrix()
        elif option == "İstatistiksel Bilgiler":
            self.show_statistical_infos()
        elif option == "Info":
            self.show_dataset_info()
        elif option == "Aykırı Verileri Ayıkla":
            self.detect_and_remove_outliers()
        elif option == "Unique":
            self.show_unique()
        elif option == "N-Unique":
            self.show_unique_number()
    
    def show_raw_data(self):
        self.clear_tree()
        self.tree["column"] = list(self.data.columns)
        self.tree["show"] = "headings"

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)

        for index, row in self.data.head().iterrows():
            self.tree.insert("", "end", values=list(row))

    def fill_na(self):
        def fill_method():
            choice = fill_choice.get()
            if choice == "Ortalama":
                for column in self.data.select_dtypes(include=["number"]).columns:
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
            elif choice == "Medyan":
                for column in self.data.select_dtypes(include=["number"]).columns:
                    self.data[column].fillna(self.data[column].median(), inplace=True)
            for column in self.data.select_dtypes(include=["object"]).columns:
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            messagebox.showinfo("Başarılı", "Boş veriler dolduruldu.")
            self.show_null_values()
            fill_window.destroy()

        fill_window = tk.Toplevel(self.root)
        fill_window.title("Boş Verileri Doldur")
        fill_window.geometry("300x100")

        tk.Label(fill_window, text="Numerik veriler için doldurma yöntemi:").pack(pady=5)
        fill_choice = tk.StringVar(value="Ortalama")
        tk.Radiobutton(fill_window, text="Ortalama", variable=fill_choice, value="Ortalama").pack(anchor='w')
        tk.Radiobutton(fill_window, text="Medyan", variable=fill_choice, value="Medyan").pack(anchor='w')

        tk.Button(fill_window, text="Doldur", command=fill_method, cursor="hand2").pack(pady=5)
    
    def detect_and_remove_outliers(self):
        outlier_indices = []
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numerical_features:
            Q1 = np.percentile(self.data[feature], 25)
            Q3 = np.percentile(self.data[feature], 75)
            IQR = Q3 - Q1
            
            outlier_step = IQR * 1.5
            
            outlier_list_col = self.data[(self.data[feature] < Q1 - outlier_step) | (self.data[feature] > Q3 + outlier_step)].index
            
            outlier_indices.extend(outlier_list_col)
        
        outlier_indices = Counter(outlier_indices)
        
        multiple_outliers = [idx for idx, count in outlier_indices.items() if count > 2]
        
        self.data = self.data.drop(multiple_outliers, axis=0).reset_index(drop=True)
        messagebox.showinfo("Başarılı", "Aykırı Değerler Silindi!")
        self.show_raw_data()
    
    def show_in_text_widget(self, text):
        self.clear_tree()
        self.tree["column"] = ["Info"]
        self.tree["show"] = "headings"
        self.tree.heading("Info", text="Info")
        self.tree.insert("", "end", values=[text])

    def show_columns(self): 
        option = self.analysis_var.get()
        self.clear_tree()
        self.tree["column"] = ["Columns"]
        self.tree["show"] = "headings"
        self.tree.heading("Columns", text="Columns")
        for col in self.data.columns:
            self.tree.insert("", "end", values=[col])
        
        if option ==  "Columnlar":
            self.tree.bind("<Button-3>", self.show_context_menu)
    
    def show_context_menu(self, event):
        self.context_menu = Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Delete", command=lambda: self.delete_column(event))
        self.context_menu.add_command(label="Split",command=lambda:self.split_column(event))
        self.context_menu.add_command(label="Convert Dtype",command=lambda:self.convertDType(event))
        self.context_menu.post(event.x_root, event.y_root)

    def delete_column(self, event):
        item = self.tree.identify_row(event.y)
        column_name = self.tree.item(item, "values")[0]
        self.data.drop(columns=[column_name], inplace=True)
        self.show_columns()
    
    def split_column(self,event):
        separator = tk.simpledialog.askstring("Input", "Enter the character to split by:", parent=self.root)
        if not separator:
            return
        item = self.tree.identify_row(event.y)
        column_name = self.tree.item(item, "values")[0]
        self.data[column_name] = self.data[column_name].apply(lambda x: self.handle_split(x, separator))
        self.show_columns()
        messagebox.showinfo("Başarılı", f"'{separator}'e göre ayrılma işlemi gerçekleştiridi")

    def handle_split(self, value, separator):
        value = str(value).split(separator)
        return value[0]  

    def convertDType(self,event):
        type = tk.simpledialog.askstring("Input", "Enter the type you want to convert:", parent=self.root)
        if not type:
            return
        item = self.tree.identify_row(event.y)
        column_name = self.tree.item(item, "values")[0]
        self.data[column_name] = self.data[column_name].apply(lambda x: self.handle_convert(x,type))
        self.show_data_types()
        messagebox.showinfo("Başarılı", f"Veri tipi {type} olarak kaydedildi")

    def handle_convert(self, value, data_type):
        try:
            if data_type == "int":
                return int(value)
            elif data_type == "float":
                return float(value)
            elif data_type == "str":
                return str(value)
            elif data_type == "bool":
                return bool(value)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
        except ValueError as e:
            print(f"Error converting value {value} to {data_type}: {e}")
            return None  
    
    def show_data_types(self):
        self.clear_tree()
        self.tree["column"] = ["Column", "Data Type"]
        self.tree["show"] = "headings"
        self.tree.heading("Column", text="Column")
        self.tree.heading("Data Type", text="Data Type")
        for col, dtype in self.data.dtypes.items():
            self.tree.insert("", "end", values=[col, dtype])
    
    def show_dataset_info(self):
        self.clear_tree()
        self.tree["columns"] = ["Column", "Info"]
        self.tree["show"] = "headings"
        self.tree.heading("Column", text="Column")
        self.tree.heading("Info", text="Info")

        buffer = io.StringIO()
        sys.stdout = buffer
        self.data.info()
        sys.stdout = sys.__stdout__

        info_lines = buffer.getvalue().splitlines()
        
        for line in info_lines[1:]:
            col_info = line.split(maxsplit=1)
            if len(col_info) == 2:
                col, info = col_info
                self.tree.insert("", "end", values=[col.strip(), info.strip()])
            else:
                self.tree.insert("", "end", values=["", line.strip()])

    def show_null_values(self):
        self.clear_tree()
        self.tree["column"] = ["Column", "Null Values"]
        self.tree["show"] = "headings"
        self.tree.heading("Column", text="Column")
        self.tree.heading("Null Values", text="Null Values")
        for col, null_count in self.data.isnull().sum().items():
            self.tree.insert("", "end", values=[col, null_count])
    
    def show_statistical_infos(self):
        self.clear_tree()
        stats = self.data.describe().T
        self.tree["column"] = ["Columns"] + stats.columns.tolist()
        self.tree["show"] = "headings"

        self.tree.heading("Columns", text="Columns")
        for col in stats.columns:
            self.tree.heading(col, text=col)

        for stat, values in stats.iterrows():
            self.tree.insert("", "end", values=[stat] + values.tolist())

    def show_corr_matrix(self):
        self.clear_tree()
        cm = self.data.corr(numeric_only=True)
        self.tree["column"] = list(cm.columns)
        self.tree["show"] = "headings"

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)

        for index, row in cm.iterrows():
            self.tree.insert("", "end", values=list(row))
        
        g = sns.clustermap(cm, 
                           method='complete', 
                           cmap='RdBu', 
                           annot=True, 
                           annot_kws={'size': 8})
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60)
        plt.show()            
   
        self.clear_tree()
        stat_infos = self.data.describe().T

        self.tree["column"] = ["name"] + list(stat_infos.columns)
        self.tree["show"] = "headings"

        for col in self.tree["column"]:
            self.tree.heading(col, text=col)

        # Insert the statistical information values into the Treeview
        for index, row in stat_infos.iterrows():
            values = [index] + list(row)
            self.tree.insert("", "end", values=values)

    def show_unique(self):
        self.clear_tree()
        self.unique_data = {col: self.data[col].unique() for col in self.data.columns}
        self.tree["columns"] = ["Columns", "Unique Values"]
        self.tree["show"] = "headings"

        self.tree.heading("Columns", text="Columns")
        self.tree.heading("Unique Values", text="Unique Values")

        for col, values in self.unique_data.items():
            limited_values_str = ", ".join(map(str, values[:5])) + ("..." if len(values) > 5 else "")
            self.tree.insert("", "end", values=[col, limited_values_str])

    def show_unique_number(self):
        self.clear_tree()
        unique_counts = self.data.nunique()
        self.tree["columns"] = ["Columns", "Number of Unique Values"]
        self.tree["show"] = "headings"

        self.tree.heading("Columns", text="Columns")
        self.tree.heading("Number of Unique Values", text="Number of Unique Values")

        for col, count in unique_counts.items():
            self.tree.insert("", "end", values=[col, count])
    
    def on_double_click_unique(self, event):
        option = self.analysis_var.get()
        if option != "Unique":
            return
        item = self.tree.selection()[0]
        column = self.tree.identify_column(event.x)
        col_idx = int(column.replace("#", "")) - 1
        if col_idx == 1:  # Unique Values column
            col_name = self.tree.item(item, "values")[0]
            full_values = self.unique_data[col_name]
            full_values_str = ", ".join(map(str, full_values))
            messagebox.showinfo(f"Unique Values for {col_name}", full_values_str)
    
    def execute_query(self):
        sort_column = self.sort_combobox.get()
        group_column = self.group_combobox.get()
        filter_column = self.filter_combobox.get()
        op_str = self.filter_condition_combobox.get()

        self.clear_tree()
        self.tree["columns"] = list(self.data.columns)
        self.tree["show"] = "headings"

        operators = {'>': operator.gt,'<': operator.lt,'>=': operator.ge,'<=': operator.le,'==': operator.eq,'!=': operator.ne}

        if sort_column:
            self.data = self.data.sort_values(by=sort_column)
        if group_column:
            filter_value = float(self.filter_value_input.get())
            grouped = self.data.groupby(group_column)
            op_func = operators[op_str]
            self.data = grouped.filter(lambda group: op_func(group[filter_column], filter_value).any())

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)

        for index, row in self.data.iterrows():
            self.tree.insert("", "end", values=list(row))
    
    def save_dataset(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data.to_csv(file_path, index=False)
            messagebox.showinfo("Başarılı", f"Yeni Veriseti {file_path} olarak kaydedildi.")
    #endregion

    #region FEATURE ENGINEERING
    def feature_engineering_page(self):
        self.feature_engineering_frame = tk.Frame(self.root)
        self.feature_engineering_frame.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        tk.Label(self.feature_engineering_frame, text="İşlem Türü:").grid(row=0, column=0, padx=5, pady=5)
        self.operation_type = tk.StringVar()
        operation_menu = tk.OptionMenu(self.feature_engineering_frame, self.operation_type, "Matematiksel İşlem", "Logaritma", "Sayı Karşılaştırma", "Sütun Karşılaştırma","Mod","Shift", "Date Split", command=self.update_inputs)
        operation_menu.grid(row=0, column=1, padx=5, pady=5)

        self.input_frame = tk.Frame(self.feature_engineering_frame)
        self.input_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.new_column_name_label = tk.Label(self.feature_engineering_frame, text="Yeni Sütun Adı:")
        self.new_column_name_entry = tk.Entry(self.feature_engineering_frame)

        self.create_button = tk.Button(self.feature_engineering_frame, text="Uygula", command=self.add_new_column, cursor="hand2")

    def update_inputs(self, selected_operation):
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        if selected_operation != "Date Split":
            self.new_column_name_label.grid(row=2, column=0, padx=5, pady=5)
            self.new_column_name_entry.grid(row=2, column=1, padx=5, pady=5)
            self.create_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        else:
            self.create_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        if selected_operation == "Matematiksel İşlem":
            tk.Label(self.input_frame, text="Birinci Sütun:").grid(row=0, column=0, padx=5, pady=5)
            self.column1 = tk.StringVar()
            column1_menu = tk.OptionMenu(self.input_frame, self.column1, *self.data.columns)
            column1_menu.grid(row=0, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="İkinci Sütun:").grid(row=1, column=0, padx=5, pady=5)
            self.column2 = tk.StringVar()
            column2_menu = tk.OptionMenu(self.input_frame, self.column2, *self.data.columns)
            column2_menu.grid(row=1, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="İşlem:").grid(row=2, column=0, padx=5, pady=5)
            self.operation = tk.StringVar()
            operation_menu = tk.OptionMenu(self.input_frame, self.operation, "Toplama", "Çıkarma", "Çarpma", "Bölme")
            operation_menu.grid(row=2, column=1, padx=5, pady=5)

        elif selected_operation == "Logaritma":
            tk.Label(self.input_frame, text="Sütun:").grid(row=0, column=0, padx=5, pady=5)
            self.column = tk.StringVar()
            column_menu = tk.OptionMenu(self.input_frame, self.column, *self.data.columns)
            column_menu.grid(row=0, column=1, padx=5, pady=5)

        elif selected_operation == "Sayı Karşılaştırma":
            tk.Label(self.input_frame, text="Sütun:").grid(row=0, column=0, padx=5, pady=5)
            self.compare_column = tk.StringVar()
            compare_column_menu = tk.OptionMenu(self.input_frame, self.compare_column, *self.data.columns)
            compare_column_menu.grid(row=0, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="Koşul:").grid(row=1, column=0, padx=5, pady=5)
            self.condition = tk.Entry(self.input_frame)
            self.condition.grid(row=1, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="İsimlendirme:").grid(row=2, column=0, padx=5, pady=5)
            self.labels = tk.Entry(self.input_frame)
            self.labels.grid(row=2, column=1, padx=5, pady=5)
        
        elif selected_operation == "Sütun Karşılaştırma":
            tk.Label(self.input_frame, text="Birinci Sütun:").grid(row=0, column=0, padx=5, pady=5)
            self.first_column = tk.StringVar()
            first_column_menu = tk.OptionMenu(self.input_frame, self.first_column, *self.data.columns)
            first_column_menu.grid(row=0, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="İkinci Sütun:").grid(row=1, column=0, padx=5, pady=5)
            self.second_column = tk.StringVar()
            second_column_menu = tk.OptionMenu(self.input_frame, self.second_column, *self.data.columns)
            second_column_menu.grid(row=1, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="Koşul:").grid(row=2, column=0, padx=5, pady=5)
            self.condition = tk.StringVar()
            condition_menu = tk.OptionMenu(self.input_frame, self.condition, ">", "<", ">=", "<=", "==", "!=")
            condition_menu.grid(row=2, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="Değerler :").grid(row=3, column=0, padx=5, pady=5)
            self.values = tk.Entry(self.input_frame)
            self.values.grid(row=3, column=1, padx=5, pady=5)

        elif selected_operation == "Date Split":
            tk.Label(self.input_frame, text="Sütun:").grid(row=0, column=0, padx=5, pady=5)
            self.date_column = tk.StringVar()
            date_column_menu = tk.OptionMenu(self.input_frame, self.date_column, *self.data.columns)
            date_column_menu.grid(row=0, column=1, padx=5, pady=5)

        elif selected_operation == "Shift":
            tk.Label(self.input_frame, text="Sütun:").grid(row=0, column=0, padx=5, pady=5)
            self.shift_column = tk.StringVar()
            shift_column_menu = tk.OptionMenu(self.input_frame, self.shift_column, *self.data.columns)
            shift_column_menu.grid(row=0, column=1, padx=5, pady=5)

        elif selected_operation == "Mod":
            tk.Label(self.input_frame, text="Sütun:").grid(row=0, column=0, padx=5, pady=5)
            self.mod_column = tk.StringVar()
            mod_column_menu = tk.OptionMenu(self.input_frame, self.mod_column, *self.data.columns)
            mod_column_menu.grid(row=0, column=1, padx=5, pady=5)

            tk.Label(self.input_frame, text="Bölücü:").grid(row=1, column=0, padx=5, pady=5)
            self.divisor = tk.Entry(self.input_frame)
            self.divisor.grid(row=1, column=1, padx=5, pady=5)
        
    def add_new_column(self):
        column_name = self.new_column_name_entry.get()
        selected_operation = self.operation_type.get()

        try:
            if selected_operation == "Matematiksel İşlem":
                col1 = self.column1.get()
                col2 = self.column2.get()
                operation = self.operation.get()

                if operation == "Toplama":
                    self.data[column_name] = self.data[col1] + self.data[col2]
                elif operation == "Çıkarma":
                    self.data[column_name] = self.data[col1] - self.data[col2]
                elif operation == "Çarpma":
                    self.data[column_name] = self.data[col1] * self.data[col2]
                elif operation == "Bölme":
                    self.data[column_name] = self.data[col1] / self.data[col2]

            elif selected_operation == "Logaritma":
                col = self.column.get()
                self.data[column_name] = np.log(self.data[col])

            elif selected_operation == "Sayı Karşılaştırma":
                col = self.compare_column.get()
                condition = self.condition.get()
                labels = self.labels.get().split(',')

                conditions = [eval(f"self.data['{col}'] {cond.strip()}") for cond in condition.split(';')]
                self.data[column_name] = np.select(conditions, labels)

            elif selected_operation == "Sütun Karşılaştırma":
                col1 = self.first_column.get()
                col2 = self.second_column.get()
                condition = self.condition.get()
                values = self.values.get().split(',')
                true_val, false_val = map(int, values)

                self.data[column_name] = np.where(eval(f"self.data['{col1}'] {condition} self.data['{col2}']"), true_val, false_val)

            elif selected_operation == "Date Split":
                date_col = self.date_column.get()
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                self.data[f'{date_col}_year'] = self.data[date_col].dt.year
                self.data[f'{date_col}_month'] = self.data[date_col].dt.month
                self.data[f'{date_col}_day'] = self.data[date_col].dt.day

            elif selected_operation == "Shift":
                col = self.shift_column.get()
                self.data[column_name] = self.data[col].shift(-1)

            elif selected_operation == "Mod":
                col = self.mod_column.get()
                divisor = int(self.divisor.get())
                self.data[column_name] = np.where(self.data[col] % divisor == 0, 1, 0)

            self.show_raw_data()

            if selected_operation == "Date Split":
                messagebox.showinfo("Başarılı", f"Tarih '{date_col}_year', '{date_col}_month', '{date_col}_day' adlı yeni sütunlara başarıyla ayrıldı.")
            else:
                messagebox.showinfo("Başarılı", f"'{column_name}' adlı yeni sütun başarıyla eklendi.")

        except Exception as e:
            messagebox.showerror("Hata", f"Yeni sütun eklenirken bir hata oluştu: {e}")
    #endregion

    #region DATA VISUALIZATION
    def ask_plot_details(self):

        self.figure = plt.Figure(figsize=(5,5))
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().grid(row=2, column=1, padx=10, pady=10)
    
        self.plot_details_frame = tk.Frame(self.root)
        self.plot_details_frame.grid(row=2, column=0, padx=10, pady=10, sticky='w')

        tk.Label(self.plot_details_frame, text="Grafiğin Türü:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.plot_type_var = tk.StringVar(value="Sütun")
        plot_types = ["Sütun","Yatay Bar", "Çizgi", "Box Plot", "Pasta", "Heatmap", "Pairplot", "Jointplot", "Scatter", "Histogram","Violin"]
        self.plot_type_menu = tk.OptionMenu(self.plot_details_frame, self.plot_type_var, *plot_types, command=self.update_parameters)
        self.plot_type_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.parameters_frame = tk.Frame(self.plot_details_frame)
        self.parameters_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        plot_button = tk.Button(self.plot_details_frame, text="Grafiği Çiz", command=self.draw_plot, cursor="hand2")
        plot_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

        self.update_parameters()

    def update_parameters(self, event=None):
        for widget in self.parameters_frame.winfo_children():
            widget.destroy()

        plot_type = self.plot_type_var.get()

        if plot_type == "Sütun":
            self.add_bar_type_selector()
        elif plot_type == "Yatay Bar":
            self.add_hor_bar_type_selector()
        elif plot_type == "Box Plot":
            self.add_box_plot_type_selector()
        elif plot_type == "Pairplot":
            self.add_pairplot_type_selector()
        elif plot_type == "Jointplot":
            self.add_jointplot_type_selector()
        elif plot_type == "Scatter":
            self.add_scatter_type_selector()
        elif plot_type == "Çizgi":
            self.add_lineplot_type_selector()
        elif plot_type == "Violin":
            self.add_violin_type_selector()
        else:
            self.add_x_column_selector()
            if plot_type in ["Heatmap", "Jointplot"]:
                self.add_y_column_selector()
    
    #region PLOT_PARAMATERS_UPDATE
    def add_x_column_selector(self):
        tk.Label(self.parameters_frame, text="X Ekseni:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.x_axis_var = tk.StringVar()
        self.x_axis_menu = tk.OptionMenu(self.parameters_frame, self.x_axis_var, *self.data.columns.tolist())
        self.x_axis_menu.grid(row=3, column=1, padx=5, pady=5, sticky='w')

    def add_y_column_selector(self, optional=False):
        label_text = "Y Ekseni:" if not optional else "Y Ekseni:"
        tk.Label(self.parameters_frame, text=label_text).grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.y_axis_var = tk.StringVar()
        self.y_axis_menu = tk.OptionMenu(self.parameters_frame, self.y_axis_var, *self.data.columns.tolist())
        self.y_axis_menu.grid(row=4, column=1, padx=5, pady=5, sticky='w')

    def add_hue_selector(self, optional=False):
        label_text = "Hue:" if not optional else "Hue:"
        tk.Label(self.parameters_frame, text=label_text).grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.hue_var = tk.StringVar()
        self.hue_menu = tk.OptionMenu(self.parameters_frame, self.hue_var, *self.data.columns.tolist())
        self.hue_menu.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        
    def add_lineplot_type_selector(self):
        tk.Label(self.parameters_frame, text="Çizgi Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.line_type_var = tk.StringVar(value="Huesuz")
        line_types = ["Huesuz", "Huelu"]
        self.line_type_menu = tk.OptionMenu(self.parameters_frame, self.line_type_var, *line_types, command=self.update_line_parameters)
        self.line_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.update_line_parameters()
    
    def update_line_parameters(self, event=None):
        for widget in self.parameters_frame.winfo_children():
            if isinstance(widget, tk.OptionMenu) and widget != self.line_type_menu:
                widget.destroy()

        line_type = self.line_type_var.get()
        self.add_x_column_selector()
        self.add_y_column_selector()

        if line_type == "Huelu":
            self.add_hue_selector()

    def add_scatter_type_selector(self):
        tk.Label(self.parameters_frame, text="Scatter Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.scatter_type_var = tk.StringVar(value="Huesuz")
        scatter_types = ["Huesuz", "Huelu"]
        self.scatter_type_menu = tk.OptionMenu(self.parameters_frame, self.scatter_type_var, *scatter_types, command=self.update_scatter_parameters)
        self.scatter_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.update_scatter_parameters()
    
    def update_scatter_parameters(self, event=None):
        for widget in self.parameters_frame.winfo_children():
            if isinstance(widget, tk.OptionMenu) and widget != self.scatter_type_menu:
                widget.destroy()

        scatter_type = self.scatter_type_var.get()
        self.add_x_column_selector()
        self.add_y_column_selector()

        if scatter_type == "Huelu":
            self.add_hue_selector()

    def add_bar_type_selector(self):
        tk.Label(self.parameters_frame, text="Bar Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.bar_type_var = tk.StringVar(value="Tek parametreli")
        bar_types = ["Tek parametreli", "İki parametreli"]
        self.bar_type_menu = tk.OptionMenu(self.parameters_frame, self.bar_type_var, *bar_types, command=self.update_bar_parameters)
        self.bar_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.update_bar_parameters()
    
    def update_bar_parameters(self, event=None):
        for widget in self.parameters_frame.winfo_children():
            if isinstance(widget, tk.OptionMenu) and widget != self.bar_type_menu:
                widget.destroy()

        bar_type = self.bar_type_var.get()
        self.add_x_column_selector()
        if bar_type == "İki parametreli":
            self.add_y_column_selector()
    
    def add_hor_bar_type_selector(self,event=None):
        tk.Label(self.parameters_frame, text="Yatay Bar Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.hor_bar_type_var = tk.StringVar(value="Tek parametreli")
        hor_bar_types = ["Tek parametreli", "İki parametreli"]
        self.hor_bar_type_menu = tk.OptionMenu(self.parameters_frame, self.bar_type_var, *hor_bar_types, command=self.update_hor_bar_parameters)
        self.hor_bar_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.update_hor_bar_parameters()
    
    def update_hor_bar_parameters(self,event=None):
        for widget in self.parameters_frame.winfo_children():
            if isinstance(widget, tk.OptionMenu) and widget != self.hor_bar_type_menu:
                widget.destroy()

        hor_bar_type = self.bar_type_var.get()
        self.add_x_column_selector()
        if hor_bar_type == "İki parametreli":
            self.add_y_column_selector()

    def add_box_plot_type_selector(self):
        tk.Label(self.parameters_frame, text="Box Plot Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.box_plot_type_var = tk.StringVar(value="Tek parametreli")
        box_plot_types = ["Tek parametreli", "İki parametreli"]
        self.box_plot_type_menu = tk.OptionMenu(self.parameters_frame, self.box_plot_type_var, *box_plot_types, command=self.update_box_plot_parameters)
        self.box_plot_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.update_box_plot_parameters()

    def update_box_plot_parameters(self, event=None):
        for widget in self.parameters_frame.winfo_children():
            if isinstance(widget, tk.OptionMenu) and widget != self.box_plot_type_menu:
                widget.destroy()

        box_plot_type = self.box_plot_type_var.get()
        self.add_x_column_selector()
        if box_plot_type == "İki parametreli":
            self.add_y_column_selector()
     
    def add_jointplot_type_selector(self):
        tk.Label(self.parameters_frame,text="JointPlot Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.jointplot_type_var = tk.StringVar(value="hist")
        jointplot_types = ["hist","hex","kde","reg","resid","scatter"]
        self.jointplot_type_menu = tk.OptionMenu(self.parameters_frame,self.jointplot_type_var,*jointplot_types)
        self.jointplot_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.add_x_column_selector()
        self.add_y_column_selector()

    def add_pairplot_type_selector(self):
        tk.Label(self.parameters_frame, text="Pairplot Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.pairplot_type_var = tk.StringVar(value="Huesuz")
        pairplot_types = ["Huesuz", "Huelu"]
        self.pairplot_type_menu = tk.OptionMenu(self.parameters_frame, self.pairplot_type_var, *pairplot_types, command=self.update_pairplot_parameters)
        self.pairplot_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.update_pairplot_parameters()

    def update_pairplot_parameters(self, event=None):
        for widget in self.parameters_frame.winfo_children():
            if isinstance(widget, tk.OptionMenu) and widget != self.pairplot_type_menu:
                widget.destroy()

        pairplot_type = self.pairplot_type_var.get()
        if pairplot_type == "Huelu":
            self.add_x_column_selector()
    
    def add_violin_type_selector(self):
        tk.Label(self.parameters_frame, text="Violin Plot Türü:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.violin_plot_type_var = tk.StringVar(value="Tek parametreli")
        violin_plot_types = ["Tek parametreli", "İki parametreli Huesuz","İki parametreli Huelu"]
        self.violin_plot_type_menu = tk.OptionMenu(self.parameters_frame, self.violin_plot_type_var, *violin_plot_types, command=self.update_violin_plot_parameters)
        self.violin_plot_type_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.update_violin_plot_parameters()
    
    def update_violin_plot_parameters(self,event=None):
        for widget in self.parameters_frame.winfo_children():
            if isinstance(widget, tk.OptionMenu) and widget != self.violin_plot_type_menu:
                widget.destroy()

        violin_plot_type = self.violin_plot_type_var.get()
        self.add_x_column_selector()
        if violin_plot_type == "İki parametreli Huesuz":
            self.add_y_column_selector()
        if violin_plot_type == "İki parametreli Huelu":
            self.add_y_column_selector()
            self.add_hue_selector()

    # endregion
    
    def draw_plot(self):
        
        plot_type = self.plot_type_var.get()

        pairplot_type = self.pairplot_type_var.get() if hasattr(self, 'pairplot_type_var') else None

        if plot_type == "Pairplot" and pairplot_type == "Huesuz":
            self.is_have_x_axis = False
        else:
            self.is_have_x_axis = True

        x_axis_column = None
        if self.is_have_x_axis:
            if self.x_axis_var.get() is None:
                messagebox.showerror("Eksik Girdi", "X Ekseni Giriniz")
                return
            x_axis_column = self.x_axis_var.get()
        

        bar_type = self.bar_type_var.get() if hasattr(self, 'bar_type_var') else None
        hor_bar_type = self.hor_bar_type_var.get() if hasattr(self, 'hor_bar_type_var') else None
        box_plot_type = self.box_plot_type_var.get() if hasattr(self, 'box_plot_type_var') else None
        scatter_plot_type = self.scatter_type_var.get() if hasattr(self,"scatter_type_var") else None
        line_plot_type = self.line_type_var.get() if hasattr(self,"line_type_var") else None
        violin_plot_type = self.violin_plot_type_var.get() if hasattr(self,"violin_plot_type_var") else None
        
        hue_value = None
        if (plot_type == "Scatter" and scatter_plot_type == "Huelu") or (plot_type == "Çizgi" and line_plot_type=="Huelu") or ((plot_type == "Violin" and violin_plot_type=="İki parametreli Huelu")):
            hue_value = self.hue_var.get()
        

        if plot_type in ["Histogram", "Pasta","Pairplot"]:
            self.is_have_y_axis = False
        elif plot_type == "Sütun" and bar_type == "Tek parametreli":
            self.is_have_y_axis = False
        elif plot_type == "Yatay Bar" and hor_bar_type == "Tek parametreli":
            self.is_have_y_axis = False
        elif plot_type == "Box Plot" and box_plot_type == "Tek parametreli":
            self.is_have_y_axis = False
        elif plot_type == "Violin" and violin_plot_type == "Tek parametreli":
            self.is_have_y_axis = False
        else:
            self.is_have_y_axis = True
        
        y_axis_column = None
        if self.is_have_y_axis:
            if self.y_axis_var.get() is None:
                messagebox.showerror("Eksik Girdi", "Y Ekseni Giriniz")
                return
            y_axis_column = self.y_axis_var.get()

        self.loading_label = tk.Label(self.root, text="Loading...")
        self.loading_label.grid(row=1, column=0, columnspan=2, pady=5)

        def update_plot():
            try:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                self.plot_graph(ax,x_axis_column, y_axis_column,hue_value, plot_type, bar_type, hor_bar_type, box_plot_type,scatter_plot_type,line_plot_type,violin_plot_type)
                self.canvas.draw()
                self.loading_label.destroy()
            except Exception as e:
                self.loading_label.destroy()
                print(e)
                messagebox.showerror("Bir sorun oluştu", f"Bir hata oluştu: Parametreleri Kontrol Ediniz")

        self.root.after(100, update_plot)

        def on_double_click(event):
            new_fig, new_ax = plt.subplots(figsize=(10, 6))
            self.plot_graph(new_ax, x_axis_column, y_axis_column,hue_value, plot_type, bar_type, hor_bar_type, box_plot_type,scatter_plot_type,line_plot_type,violin_plot_type)
            if plot_type != "Jointplot": 
                plt.show()

        self.canvas.get_tk_widget().bind("<Double-1>", on_double_click)

    def plot_graph(self, ax, x_axis_column=None, y_axis_column=None,hue_value=None, plot_type=None, bar_type=None, hor_bar_type=None, box_plot_type=None,scatter_plot_type=None,line_plot_type=None,violin_plot_type=None):
        style = "whitegrid"
        if plot_type == "Sütun":
            if bar_type == "Tek parametreli":
                self.data[x_axis_column].value_counts().plot(kind='bar', ax=ax)
                ax.set_xlabel(x_axis_column)
                ax.set_ylabel("Value Count")
                ax.set_title(f"{x_axis_column} Value Count Sütun Grafiği")
            else:
                plot_data = self.data.groupby([x_axis_column, y_axis_column]).size().unstack()
                plot_data.plot(kind="bar", stacked=False, ax=ax)
                ax.set_xlabel(x_axis_column)
                ax.set_ylabel(y_axis_column)
                ax.set_title(f"{x_axis_column}-{y_axis_column} Sütun Grafiği")
        elif plot_type == "Yatay Bar":
            if hor_bar_type == "Tek parametreli":
                data_count = self.data[x_axis_column].value_counts()
                sns.set_style(style)
                sns.barplot(y=data_count.index, x=data_count.values,orient="h", ax=ax)
                ax.set_title(f'{x_axis_column} distribution')
                ax.set_xlabel('no')
                ax.set_ylabel(f'{x_axis_column}')
            else:
                data_worth = self.data.groupby(x_axis_column)[y_axis_column].mean().sort_values()
                sns.set_style(style)
                sns.barplot(y=data_worth.index, x=data_worth.values,orient="h", ax=ax)
                ax.set_title(f'Average {y_axis_column} by {x_axis_column}')
                ax.set_xlabel(f'Average {y_axis_column}')
                ax.set_ylabel(f'{x_axis_column}')
        elif plot_type == "Çizgi":
            # grouped_data = self.data.groupby(x_axis_column)[y_axis_column].sum()
            # result = pd.DataFrame({y_axis_column: grouped_data})
            if line_plot_type == "Huesuz":
                sns.set_style(style)
                sns.lineplot(data=self.data,x=x_axis_column,y=y_axis_column,ax=ax)
                ax.set_xlabel(x_axis_column)
                ax.set_ylabel(y_axis_column)
                ax.set_title(f" {x_axis_column}-{y_axis_column}  Çizgi Grafiği")
            else:
                sns.set_style(style)
                sns.lineplot(data=self.data,x=x_axis_column,y=y_axis_column,hue=hue_value,ax=ax)
                ax.set_xlabel(x_axis_column)
                ax.set_ylabel(y_axis_column)
                ax.set_title(f" {x_axis_column}-{y_axis_column}  Çizgi Grafiği with hue {hue_value}")
        elif plot_type == "Box Plot":
            if box_plot_type == "Tek parametreli":
                sns.set_style(style)
                sns.boxplot(x=self.data[x_axis_column], ax=ax, orient='h')
                ax.set_title(f"{x_axis_column} Box Plot")
            else:
                sns.set_style(style)
                sns.boxplot(x=x_axis_column, y=y_axis_column, data=self.data, ax=ax, orient='h')
                ax.set_title(f"{x_axis_column}-{y_axis_column} Box Plot")
        elif plot_type == "Violin":
            if violin_plot_type == "Tek parametreli":
                sns.set_style(style)
                sns.violinplot(y=self.data[x_axis_column],ax=ax)
                ax.set_title(f"{x_axis_column} Violin Plot")
            elif violin_plot_type == "İki parametreli Huesuz":
                sns.set_style(style)
                sns.violinplot(data=self.data, x=x_axis_column, y=y_axis_column,ax=ax)
                ax.set_title(f"{x_axis_column}-{y_axis_column} Violin Plot")
            else:
                sns.set_style(style)
                sns.violinplot(data=self.data, x=x_axis_column, y=y_axis_column,hue=hue_value,ax=ax)
                ax.set_title(f"{x_axis_column}-{y_axis_column} Violin Plot with Hue {hue_value}")
        elif plot_type == "Heatmap":
            pivot_table = self.data.pivot_table(index=y_axis_column, columns=x_axis_column, aggfunc='size', fill_value=0)
            sns.set_style(style)
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt='d', ax=ax)
            ax.set_title(f"{x_axis_column}-{y_axis_column} Heatmap")
        elif plot_type == "Pasta":
                self.data[x_axis_column].value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
                ax.set_ylabel(x_axis_column)
                ax.set_title("Pasta Grafiği")
        elif plot_type == "Pairplot":
            if self.pairplot_type_var.get() == "Huelu":
                sns.set_style(style)
                sns.pairplot(self.data, hue=x_axis_column)
                ax.set_title(f" Pairplot with Hue {x_axis_column}")
                plt.show()
            else:
                sns.set_style(style)
                sns.pairplot(self.data)
                ax.set_title(f"Pairplot")
                plt.show()
        elif plot_type == "Jointplot":
            kind = self.jointplot_type_var.get()
            sns.set_style(style)
            sns.jointplot(x=x_axis_column, y=y_axis_column, data=self.data,kind=kind)
            ax.set_title("Jointplot")
            plt.show()
        elif plot_type == "Scatter":
            if scatter_plot_type == "Huesuz":
                sns.set_style(style)
                sns.scatterplot(data=self.data, x=x_axis_column, y=y_axis_column, ax=ax)
                ax.set_xlabel(x_axis_column)
                ax.set_ylabel(y_axis_column)
                ax.set_title(f"{x_axis_column}-{y_axis_column} Scatter Grafiği")
            else:
                sns.set_style(style)
                sns.scatterplot(data=self.data, x=x_axis_column, y=y_axis_column, hue=hue_value, ax=ax)
                ax.set_xlabel(x_axis_column)
                ax.set_ylabel(y_axis_column)
                ax.set_title(f"{x_axis_column}-{y_axis_column} Scatter Grafiği with Hue {hue_value}")
        elif plot_type == "Histogram":
            sorted_data = self.data[x_axis_column].sort_values()
            sns.set_style(style)
            sns.histplot(sorted_data, kde=True, bins=30, ax=ax)
            ax.set_xlabel(x_axis_column)
            ax.set_ylabel("Frekans")
            ax.set_title(f"{x_axis_column} Histogram")
    #endregion

    #region MODEL TRAINING
    def ask_model_details(self):
        self.model_details_frame = tk.Frame(self.root)
        self.model_details_frame.grid(row=3, column=0, padx=10, pady=10, sticky='w')
         
        tk.Label(self.model_details_frame, text="İşem Tipi:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.proccess_category_var = tk.StringVar(value="Reg")
        procces_types = ["Reg","Class","Clust"]
        self.procces_category_menu= ttk.Combobox(self.model_details_frame, textvariable=self.proccess_category_var,values=procces_types, cursor="hand2")
        self.procces_category_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        tk.Button(self.model_details_frame,text="İşlem Seç",command=self.update_model_types,cursor="hand2").grid(row=0, column=2, padx=5, pady=5, sticky='w')
    
    def update_model_types(self,event=None):
        model_types_dict = {"Reg":["Lineer Regresyon","Polinomal Regresyon","Lojistik Regresyon","SVR","Desicion Tree Regressor","Random Forest Regressor"],
                            "Class":["KNN Classifier","SVC","Naive Bayes Classifier","Desicion Tree Classifier","Random Forest Classifier","XGB Classifier","CATBOOST Classifier","Gradiant Boost Classifier"],
                            "Clust":["KMEAN Cluster","Agglomerative Cluster","Dendrogram"]} 
        selected_proccess_type = self.proccess_category_var.get()

        tk.Label(self.model_details_frame, text="Model Tipi:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.model_category_var = tk.StringVar(value=model_types_dict[selected_proccess_type][0])
        model_types = model_types_dict[selected_proccess_type]
        self.model_category_menu= ttk.Combobox(self.model_details_frame, textvariable=self.model_category_var,values=model_types, cursor="hand2")
        self.model_category_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        tk.Button(self.model_details_frame,text="Model Seç",command=self.update_model_options,cursor="hand2").grid(row=0, column=2, padx=5, pady=5, sticky='w')

        tk.Label(self.model_details_frame, text="X Columns:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.x_columns_listbox = tk.Listbox(self.model_details_frame, selectmode=tk.MULTIPLE, height=4)
        self.x_columns_listbox.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.columns_list = []
        list=[]
        for col in self.data.columns:
            list.append(col)
            self.columns_list.append(col)
            self.x_columns_listbox.insert(tk.END, col)
        tk.Button(self.model_details_frame,text="X Ayarla",command=self.select_x,cursor="hand2").grid(row=1, column=2, padx=5, pady=5, sticky='w')

        self.y_column_label = tk.Label(self.model_details_frame, text="Y Column:")
        self.y_column_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.y_column_var = tk.StringVar(value="")
        self.y_column_menu = ttk.Combobox(self.model_details_frame, textvariable=self.y_column_var, values=list, state="readonly")
        self.y_column_menu.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        self.additional_options_frame = tk.Frame(self.model_details_frame)
        self.additional_options_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        self.additional_options_frame.grid_remove()

        self.test_size_label = tk.Label(self.model_details_frame, text="Test Yüzdesi:")
        self.test_size_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.test_size = tk.Entry(self.model_details_frame)
        self.test_size.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        
        tk.Button(self.model_details_frame,text="Fit",command=self.model_fit,cursor="hand2").grid(row=6, column=0, padx=5, pady=5, sticky='w')
    
    def update_model_options(self, event=None):
        selected_model = self.model_category_var.get()
            
        for widget in self.additional_options_frame.winfo_children():
            widget.destroy()

        if selected_model == "Polinomal Regresyon":
            tk.Label(self.additional_options_frame, text="Derece:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.degree = tk.Entry(self.additional_options_frame)
            self.degree.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        elif selected_model == "KNN Classifier":
            tk.Label(self.additional_options_frame, text="Komşu Sayısı (k):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.k = tk.Entry(self.additional_options_frame)
            self.k.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        elif selected_model in ["SVR", "SVC"]:
            tk.Label(self.additional_options_frame, text="Kernel:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.kernel = tk.Entry(self.additional_options_frame)
            self.kernel.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        elif selected_model in ["Random Forest Regressor", "Random Forest Classifier"]:
            tk.Label(self.additional_options_frame, text="N-Estimator:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.n_estimator = tk.Entry(self.additional_options_frame)
            self.n_estimator.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        elif selected_model == "CATBOOST Classifier" :
            tk.Label(self.additional_options_frame, text="Iterations:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.iterations = tk.Entry(self.additional_options_frame)
            self.iterations.grid(row=0, column=1, padx=5, pady=5, sticky='w')
            tk.Label(self.additional_options_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
            self.lr = tk.Entry(self.additional_options_frame)
            self.lr.grid(row=1, column=1, padx=5, pady=5, sticky='w')
            tk.Label(self.additional_options_frame, text="Depth:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
            self.depth = tk.Entry(self.additional_options_frame)
            self.depth.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        elif selected_model == "KMEAN Cluster":
            tk.Label(self.additional_options_frame, text="N-Clusters:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.n_clusters = tk.Entry(self.additional_options_frame)
            self.n_clusters.grid(row=0, column=1, padx=5, pady=5, sticky='w')
            self.y_column_label.grid_remove()
            self.test_size_label.grid_remove()
            self.y_column_menu.grid_remove()
            self.test_size.grid_remove()
        elif selected_model == "Agglomerative Cluster":
            tk.Label(self.additional_options_frame, text="N-Clusters:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.n_clusters = tk.Entry(self.additional_options_frame)
            self.n_clusters.grid(row=0, column=1, padx=5, pady=5, sticky='w')
            self.y_column_label.grid_remove()
            self.test_size_label.grid_remove()
            self.y_column_menu.grid_remove()
            self.test_size.grid_remove()
        elif selected_model == "Dendrogram":
            self.y_column_label.grid_remove()
            self.test_size_label.grid_remove()
            self.y_column_menu.grid_remove()
            self.test_size.grid_remove()
        self.additional_options_frame.grid()
    
    def select_x(self):
        self.selected_x_indices = self.x_columns_listbox.curselection()
        self.selected_x_columns = [self.columns_list[i] for i in self.selected_x_indices]

    def model_fit(self): 
        try:
            kernels = ["linear","poly","rbf","sigmoid","precomputed"]
            isCluster = False
            scaler = StandardScaler()
            model_type = self.model_category_menu.get()
            self.status_label = tk.Label(self.model_details_frame,text="Eğitim Başlıyor ...")
            self.status_label.grid(row=7,column=0,padx=5,pady=5,sticky="w")
            if self.selected_x_columns is None:
                messagebox.showerror("Eksik Değer","x değerlerini girmelisiniz")
                return
            x_df = self.data[self.selected_x_columns]
            x_df = pd.get_dummies(x_df)
            x_df = scaler.fit_transform(x_df)

            metrics = ""

            if model_type not in ["Agglomerative Cluster","KMEAN Cluster","Dendrogram"]:
                if self.y_column_menu.get() is None:
                    messagebox.showerror("Eksik Değer","y değerini girmelisiniz")
                    return
                selected_y_column = self.y_column_menu.get()
                y_df = self.data[selected_y_column]
                if y_df.dtype == 'object':
                    label_encoder = LabelEncoder()
                    y_df = label_encoder.fit_transform(y_df)
                if self.test_size.get() is None:
                    messagebox.showerror("Eksik Değer","test size değerini girmelisiniz")
                    return
                if int(self.test_size.get())<0 and int(self.test_size.get())>100:
                    messagebox.showerror("Hatalı Değer","test size değeri 0 ile 100 arasında olmalı")
                    return
                test_size = float(self.test_size.get())/100
                xtrain,xtest,ytrain,ytest=train_test_split(x_df,y_df,test_size=test_size,random_state=114)

            if model_type == "Lineer Regresyon":
                self.model = LinearRegression()
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Lineer Regresyon",True)
            elif model_type == "Polinomal Regresyon":
                if self.degree.get() is None:
                    messagebox.showerror("Eksik Değer","Derece değreri giriniz")
                    return
                degree = int(self.degree.get())
                poly_features = PolynomialFeatures(degree=degree)
                X_train_poly = poly_features.fit_transform(xtrain)
                X_test_poly = poly_features.transform(xtest)
                self.model = LinearRegression()
                metrics = self.model_fit_metrics_helper(self.model,X_train_poly,X_test_poly,ytrain,ytest,"Polinomal Regresyon",True)
            elif model_type == "Lojistik Regresyon":
                self.model = LogisticRegression()
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Lojistik Regresyon",False)
            elif model_type == "SVR":
                if self.kernel.get() is None:
                    messagebox.showerror("Eksik Değer","Kernel değreri giriniz")
                    return
                if self.kernel.get() not in kernels:
                    messagebox.showerror("Hatalı Değer",f"'{self.kernel.get()}' geçerli bir kernel değil")
                    return
                kernel = self.kernel.get()
                self.model = SVR(kernel=kernel, random_state=114)
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"SVR",True)
            elif model_type == "Desicion Tree Regressor":
                self.model = DecisionTreeRegressor(random_state=114)
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Desicion Tree Regressor",True)
            elif model_type == "Random Forest Regressor":
                n_estimators = int(self.n_estimator.get())
                self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=114)
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Random Forest Regressor",True)
            elif model_type == "KNN Classifier":
                if self.k.get() is None:
                    messagebox.showerror("Eksik Değer","K değreri giriniz")
                    return
                k = int(self.k.get())
                self.model = KNeighborsClassifier(n_neighbors=k)
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"KNN Classifier",False)
                print(metrics)
            elif model_type == "SVC":
                if self.kernel.get() is None:
                    messagebox.showerror("Eksik Değer","Kernel değreri giriniz")
                    return
                if self.kernel.get() not in kernels:
                    messagebox.showerror("Hatalı Değer",f"'{self.kernel.get()}' geçerli bir kernel değil  ")
                    return
                kernel = self.kernel.get()
                self.model = SVC(kernel=kernel, random_state=114)
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"SVC",False)
            elif model_type == "Naive Bayes Classifier":
                self.model=GaussianNB()
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Naive Bayes Classifier",False)
            elif model_type ==  "Desicion Tree Classifier":
                self.model = DecisionTreeClassifier(random_state=114,criterion="entropy")
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Desicion Tree Classifier",False)
            elif model_type == "Random Forest Classifier":
                if self.n_estimator.get() is None:
                    messagebox.showerror("Eksik Değer","N-Estimator değreri giriniz")
                    return
                n_estimators = int(self.n_estimator.get())
                self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=114,criterion="entropy")
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Random Forest Classifier",False)
            elif model_type == "XGB Classifier":
                self.model = XGBClassifier()
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"XGB Classifier",False)
            elif model_type == "CATBOOST Classifier":
                if self.iterations.get() is None:
                    messagebox.showerror("Eksik Değer","Iterasyon değreri giriniz")
                    return
                if self.lr.get() is None:
                    messagebox.showerror("Eksik Değer","LR değreri giriniz")
                    return
                if self.depth.get() is None:
                    messagebox.showerror("Eksik Değer","Depth değreri giriniz")
                    return
                iterations = int(self.iterations.get())
                lr=int(self.lr.get())
                depth=int(self.depth.get())
                self.model = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=depth, verbose=False)
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"CATBOOST Classifier",False)
            elif model_type == "Gradiant Boost Classifier":
                self.model = GradientBoostingClassifier()
                metrics = self.model_fit_metrics_helper(self.model,xtrain,xtest,ytrain,ytest,"Gradiant Boost Classifier",False)
            elif model_type == "KMEAN Cluster":
                isCluster = True
                if self.n_clusters.get() is None:
                    messagebox.showerror("Eksik Değer","N-cluster değreri giriniz")
                    return
                n_clusters = int(self.n_clusters.get())
                self.model = KMeans(n_clusters=n_clusters,init="k-means++" ,random_state=114)
                y_pred = self.model.fit_predict(X=x_df)
                self.plot_cluster("KMEAN Cluster",x_df,y_pred)
            elif model_type == "Agglomerative Cluster":
                isCluster = True
                n_clusters = int(self.n_clusters.get())
                self.model = AgglomerativeClustering(n_clusters=n_clusters,affinity="euclidean",linkage="ward")
                y_pred = self.model.fit_predict(X=x_df)
                self.plot_cluster("Agglomerative Cluster",x_df,y_pred)
            elif model_type == "Dendrogram":
                isCluster = True
                dendrogram = sch.dendrogram(sch.linkage(x_df,method="ward"))
                plt.show()

            self.status_label.grid_forget()
            info_text = f"Eğitim Tamamlandı!\nKümelendirme grafiğini inceleyebilirsiniz." if isCluster else f"Eğitim Tamamlandı!\n{metrics}"
            self.mse_label = tk.Label(self.model_details_frame, text=info_text)
            self.mse_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")
            self.show_save_button()
            
        except Exception as e:
            print(e.args)
            messagebox.showerror("Hata", "Parametreleri Kontrol edin")
    
    def model_fit_metrics_helper(self, model, x_train, x_test, y_train, y_test, model_name, isRegression):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)
        if isRegression:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            metrics = f"MSE: {mse}\n MAE: {mae}\n RMSE: {rmse}\n R²: {r2}"
            self.show_regression_model_details_graphs(model,y_pred,y_train_pred,x_train,x_test,y_train,y_test,model_name)
        else:
            y_prob = model.predict_proba(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            metrics = f"Accuracy: {accuracy}\n F1 Score: {f1}\n Precision: {precision}\n Recall: {recall}\n CM : {cm}"
            self.show_classification_model_details_graphs(model,x_train,x_test,y_train,y_test,y_pred,y_prob,model_name)
            #self.show_corr_matrix_heatmap(cm, model_name)

        return metrics
    
    def show_regression_model_details_graphs(self, model, y_test_pred, y_train_pred, X_train, X_test, y_train, y_test, model_type):
        if model_type == "Lineer Regresyon":
            # Kalanları hesaplama
            train_errors = y_train - y_train_pred
            test_errors = y_test - y_test_pred

            # Subplot oluşturma
            fig, axs = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle('Lineer Regresyon Modelinin Analizi', fontsize=16)

            # Eğitim verisi grafiği
            axs[0, 0].scatter(X_train, y_train, color='blue', label='Gerçek Eğitim Verileri')
            axs[0, 0].plot(X_train, y_train_pred, 'r-', label='Tahmin Edilen Eğitim Verileri')
            axs[0, 0].set_title('Eğitim Verileri: Öngörü vs. Gerçek Değerler')
            axs[0, 0].legend()

            # Test verisi grafiği
            axs[0, 1].scatter(X_test, y_test, color='blue', label='Gerçek Test Verileri')
            axs[0, 1].plot(X_test, y_test_pred, 'r-', label='Tahmin Edilen Test Verileri')
            axs[0, 1].set_title('Test Verileri: Öngörü vs. Gerçek Değerler')
            axs[0, 1].legend()

            # Kalan grafiği
            axs[1, 0].scatter(y_test_pred, test_errors, c='blue', label='Test Kalanları')
            axs[1, 0].axhline(0, color='red', linestyle='--')
            axs[1, 0].set_title('Test Kalanları vs. Tahmin Edilen Test Değerleri')
            axs[1, 0].set_xlabel('Tahmin Edilen Test Değerleri')
            axs[1, 0].set_ylabel('Kalanlar')

            # Hata dağılımı grafiği
            sns.histplot(test_errors, kde=True, ax=axs[1, 1])
            axs[1, 1].set_title('Test Hata Dağılımı')

            # Özniteliklerin etkisi grafiği (model katsayıları)
            coefficients = [model.intercept_] + list(model.coef_)
            features = ['Intercept'] + [f'Öznitelik {i+1}' for i in range(len(model.coef_))]
            axs[2, 0].bar(features, coefficients, color=['blue'] + ['green'] * len(model.coef_))
            axs[2, 0].set_title('Özelliklerin Etkisi')

            # Öngörü aralıkları grafiği (test seti için)
            prediction_interval = 1.96 * np.std(test_errors)  # Örnek aralıklar
            X_test_flat = X_test.flatten()  # X_test'i 1D diziye çevir
            y_test_pred_flat = y_test_pred.flatten()  # y_test_pred'i 1D diziye çevir
            axs[2, 0].plot(X_test_flat, y_test_pred_flat, 'r-', label='Tahmin Edilen Test Verileri')
            axs[2, 0].fill_between(X_test_flat, y_test_pred_flat - prediction_interval, y_test_pred_flat + prediction_interval, color='gray', alpha=0.2)
            axs[2, 0].scatter(X_test_flat, y_test, color='blue', label='Gerçek Test Verileri')
            axs[2, 0].set_title('Öngörü Aralıkları (Test Seti)')
            axs[2, 0].legend()

            # Boş subplot
            axs[2, 1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Üst başlık için boşluk bırakır
            plt.show()

        elif model_type == "Lojistik Regresyon":
            # ROC eğrisi ve AUC hesaplama
            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
            auc_score = roc_auc_score(y_test, y_test_pred)

            # Subplot oluşturma
            fig, axs = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle('Lojistik Regresyon Modelinin Analizi', fontsize=16)

            # ROC eğrisi grafiği
            axs.plot(fpr, tpr, color='blue', label=f'ROC Eğrisi (AUC = {auc_score:.2f})')
            axs.plot([0, 1], [0, 1], color='red', linestyle='--', label='Rastgele Tahmin')
            axs.set_title('ROC Eğrisi')
            axs.set_xlabel('False Positive Rate')
            axs.set_ylabel('True Positive Rate')
            axs.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        elif model_type in ["Desicion Tree Regressor", "Random Forest Regressor"]:
            n_features = X_train.shape[1]  # Özellik sayısını al

            # Subplot oluşturma
            fig, axs = plt.subplots(2, n_features, figsize=(14, 8))
            fig.suptitle(f'{model_type} Modelinin Analizi', fontsize=16)

            # Model tahmin aralıkları
            for i in range(n_features):
                feature_train = X_train[:, i] if n_features > 1 else X_train
                feature_test = X_test[:, i] if n_features > 1 else X_test

                axs[0, i].scatter(feature_train, y_train, color='blue', label='Gerçek Eğitim Verileri')
                axs[0, i].scatter(feature_train, y_train_pred, color='red', label='Tahmin Edilen Eğitim Verileri')
                axs[0, i].set_title(f'Eğitim Verileri: Gerçek vs. Tahmin (Özellik {i+1})')
                axs[0, i].legend()

                axs[1, i].scatter(feature_test, y_test, color='blue', label='Gerçek Test Verileri')
                axs[1, i].scatter(feature_test, y_test_pred, color='red', label='Tahmin Edilen Test Verileri')
                axs[1, i].set_title(f'Test Verileri: Gerçek vs. Tahmin (Özellik {i+1})')
                axs[1, i].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        elif model_type == "Polinomal Regresyon":
            # Subplot oluşturma
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('Polinomal Regresyon Modelinin Analizi', fontsize=16)

            # Eğitim verisi grafiği
            axs[0].scatter(X_train, y_train, color='blue', label='Gerçek Eğitim Verileri')
            sort_idx = np.argsort(X_train.flatten())
            axs[0].plot(X_train.flatten()[sort_idx], y_train_pred[sort_idx], 'r-', label='Tahmin Edilen Eğitim Verileri')
            axs[0].set_title('Eğitim Verileri: Öngörü vs. Gerçek Değerler')
            axs[0].legend()

            # Test verisi grafiği
            axs[1].scatter(X_test, y_test, color='blue', label='Gerçek Test Verileri')
            sort_idx = np.argsort(X_test.flatten())
            axs[1].plot(X_test.flatten()[sort_idx], y_test_pred[sort_idx], 'r-', label='Tahmin Edilen Test Verileri')
            axs[1].set_title('Test Verileri: Öngörü vs. Gerçek Değerler')
            axs[1].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        elif model_type == "SVR":
            # SVR grafikleri
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
                X_test = X_test.reshape(-1, 1)

            # Subplot oluşturma
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('SVR Modelinin Analizi', fontsize=16)

            # Eğitim verisi grafiği
            axs[0].scatter(X_train, y_train, color='blue', label='Gerçek Eğitim Verileri')
            axs[0].plot(X_train, model.predict(X_train), 'r-', label='Tahmin Edilen Eğitim Verileri')
            axs[0].set_title('Eğitim Verileri: Öngörü vs. Gerçek Değerler')
            axs[0].legend()

            # Test verisi grafiği
            axs[1].scatter(X_test, y_test, color='blue', label='Gerçek Test Verileri')
            axs[1].plot(X_test, model.predict(X_test), 'r-', label='Tahmin Edilen Test Verileri')
            axs[1].set_title('Test Verileri: Öngörü vs. Gerçek Değerler')
            axs[1].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    def show_classification_model_details_graphs(self, model, X_train, X_test, y_train, y_test, y_pred, y_prob, model_type):
        cm = confusion_matrix(y_test, y_pred)
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Karışıklık Matrisi
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
        axs[0, 0].set_title('Karışıklık Matrisi')
        axs[0, 0].set_xlabel('Tahmin Edilen Sınıf')
        axs[0, 0].set_ylabel('Gerçek Sınıf')
        
        # ROC Eğrisi
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        sns.lineplot(x=fpr, y=tpr, ax=axs[0, 1], color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        axs[0, 1].plot([0, 1], [0, 1], 'r--')
        axs[0, 1].set_title('ROC Eğrisi')
        axs[0, 1].set_xlabel('False Positive Rate')
        axs[0, 1].set_ylabel('True Positive Rate')
        
        # Precision-Recall Eğrisi
        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1], pos_label=1)
        pr_auc = auc(recall, precision)
        sns.lineplot(x=recall, y=precision, ax=axs[1, 0], color='green', label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
        axs[1, 0].set_title('Precision-Recall Eğrisi')
        axs[1, 0].set_xlabel('Recall')
        axs[1, 0].set_ylabel('Precision')
        
        # Özelliklerin Önemi (Sadece ağaç tabanlı modeller için)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            x = pd.DataFrame(X_train)
            features = x.columns
            sns.barplot(x=importances, y=features, orient="h", ax=axs[1, 1], palette='viridis')
            axs[1, 1].set_title('Özelliklerin Önemi')
        else:
            axs[1, 1].axis('off')
        
        # Öğrenme Eğrileri (Örnek)
        train_errors = np.array([model.score(X_train, y_train) for _ in range(1, len(X_train) + 1)])
        val_errors = np.array([model.score(X_test, y_test) for _ in range(1, len(X_test) + 1)])
        axs[2, 0].plot(train_errors, label='Eğitim Hatası')
        axs[2, 0].plot(val_errors, label='Doğrulama Hatası')
        axs[2, 0].set_title('Öğrenme Eğrileri')
        axs[2, 0].set_xlabel('Örnek Sayısı')
        axs[2, 0].set_ylabel('Doğruluk')
        axs[2, 0].legend()
        
        # Sınıf Dağılımı
        sns.histplot(y_test, ax=axs[2, 1], kde=False, bins=np.arange(len(np.unique(y_test)) + 1) - 0.5)
        axs[2, 1].set_title('Sınıf Dağılımı')
        axs[2, 1].set_xlabel('Sınıf')
        axs[2, 1].set_ylabel('Frekans')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Üst başlık için boşluk bırakır
        plt.show()


    def show_corr_matrix_heatmap(self,cm,model_name):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.show()
    
    def plot_cluster(self,type,X,Y_pred):
        plt.scatter(X[Y_pred==0,0] ,X[Y_pred==0,1],s=100, c='red')
        plt.scatter(X[Y_pred==1,0],X[Y_pred==1,1],s=100, c='blue')
        plt.scatter(X[Y_pred==2,0],X[Y_pred==2,1],s=100, c='green') 
        plt.scatter(X[Y_pred==3,0],X[Y_pred==3,1],s=100, c='orange') 
        plt.title(type)
        plt. show()
    
    def show_save_button(self):
        tk.Button(self.model_details_frame, text="Modeli Kaydet", command=self.save_model, cursor="hand2").grid(row=8, column=0, padx=5, pady=5, sticky='w')

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if file_path:
            try:
                joblib.dump(self.model, file_path)
                messagebox.showinfo("Başarı", f"Model başarıyla {file_path} konumuna kaydedildi.")
            except Exception as e:
                messagebox.showerror("Hata", f"Model kaydedilirken hata oluştu: {e}")
    #endregion

#region MAIN
if __name__ == "__main__":
    root = tk.Tk()
    app = MLMasterApp(root)
    root.mainloop()
#endregion        

#region TODOS

#TODO: grafiklere renkpaleti stil gibi özelleştirmeler ekle
#TODO: modele model grafikleri   model.summary ekle
#TODO: use ai seçeneği ile pandasai kullanma

#TODO: YAPILDI jointplot hatası çözümü , model traininge try except ,model save 
#TODO: YAPILDI grafik kısımlarındaki hata ayıklamalarını detaylandır
#TODO: YAPILDI preproccessinge uniqe ekle
#TODO: YAPILDI feature engineering özellikleri ekle log transform,4 işlem ile yeni columlar oluşturma,date split
#TODO: YAPILDI preprocess ve analiz edilip görselleştirilmiş veriler eğitime hazır model training sayfasında
#TODO: YAPILDI bağımsız ve bağımlı değişken sütunları seçip train test oranını gerip regression classification ya da clustering 
#TODO: YAPILDI kategorilerindeki modelleri seçip parametrelirini girip modeli eğit butonuyla eğitme ve test kısmı ekleme 
#TODO: YAPILDI preprocessing kısmına aykırı verileri ayıklama ekleme
#TODO: YAPILDI model eğitimi kısmında başta işlem türü seçsin (reg,class,clust) işlem türü seçtikten sonra model türü seçme kısmı açılsın sadece o türdekiler gözüksün
#endregion

