import pandas
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Загружаем данные и очищаем их от нулл значений(при желании можно заменить их на средние или медианные)
data_file = pandas.read_csv('files/penguins.csv')
data_file_cleaned = data_file.dropna().copy()

# Подготавливаем данные для нашей модели
# X - признаки(независимые переменные)
X = data_file_cleaned[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# y - целевая переменная
y = data_file_cleaned['species']

# Поскольку наша зависимая переменная категориальная - нам нужно ее закодировать для дальнейшего использования
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Рисуем боксы для анализа связей между нашими переменными
figure1, axes = plt.subplots(2, 2, figsize=(15, 10))
seaborn.boxplot(data=data_file_cleaned, x='species', y='bill_length_mm', ax=axes[0,0])
seaborn.boxplot(data=data_file_cleaned, x='species', y='bill_depth_mm', ax=axes[0,1])
seaborn.boxplot(data=data_file_cleaned, x='species', y='flipper_length_mm', ax=axes[1,0])
seaborn.boxplot(data=data_file_cleaned, x='species', y='body_mass_g', ax=axes[1,1])
plt.tight_layout()
plt.show()

# Разделяем данные(30%/70%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Создаем нашу модель лог. регресии и обучаем ее на тестовых данных
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X_train, y_train)

# Делаем предсказания
y_pred = model.predict(X_test)

# Выводим точность модели
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Строим матрицу ошибок
conf_matrix = confusion_matrix(y_test, y_pred)

# Визуализируем
figure2, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Матрица ошибок: Предсказания vs Фактические значения')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

