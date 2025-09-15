import pandas
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data_file = pandas.read_csv('files/tips.csv')

corr_matrix = data_file[['total_bill', 'tip', 'size']].corr()
seaborn.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Матрица корреляций')
plt.show()

# Возьмем независиммые переменные size и total_bill,так как они довольно сильно коррелирует с нашим tip
X = data_file[['total_bill', 'size']]

# Наша целевая переменная (зависимая переменная)
y = data_file['tip']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель линейной регрессии и обучаем ее на обучающих данных
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем наш прогноз на тестовых данных
y_pred = model.predict(X_test)

# Выводим коэффициенты модели (уравнение линейной регрессии y= a + b1x1 + b2x2)
print("Коэффициенты (b1, b2):", model.coef_)
print("Пересечение (a):", model.intercept_)
print("\nУравнение модели: Tip = {:.2f} + {:.2f} * total_bill + {:.2f} * size".format(model.intercept_, model.coef_[0], model.coef_[1]))

# Вычисляем качество нашей модели
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Оценка качества модели на тестовых данных:")
print(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
print(f"R² (Коэффициент детерминации): {r2:.4f}")

# Визуализируем предсказания vs Фактические значения
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Фактические чаевые (y_test)')
plt.ylabel('Предсказанные чаевые (y_pred)')
plt.title('Фактические vs Предсказанные значения')
plt.show()

new_data = pandas.DataFrame({'total_bill': [50.0], 'size': [4]})
print(f"Прогноз чаевых для счета $50 и компании из 4 человек: ${model.predict(new_data)[0]:.2f}")


