import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# تنظيف الشاشة وإعداد البيئة
plt.style.use('default')
sns.set_palette("husl")
# إزالة %matplotlib inline - لا يعمل خارج جوبيتر

print("🎯 بدء التحليل الاستكشافي للبيانات (EDA)")

# 1. الفهم الأولي للبيانات
print("\n" + "=" * 50)
print("1. الفهم الأولي للبيانات")
print("=" * 50)

# إنشاء بيانات تجريبية أولاً (لأن df غير معرّف)
np.random.seed(42)
n_records = 500
data = {
    'order_id': [f'ORD-2024-{i:04d}' for i in range(1, n_records + 1)],
    'order_date': pd.date_range('2024-01-01', periods=n_records, freq='D'),
    'customer_id': [f'CUST-{i:04d}' for i in range(1, n_records + 1)],
    'customer_name': np.random.choice(['علي محمد', 'فاطمة أحمد', 'عمر خالد', 'ليلى حسن', 'يوسف عبدالله'], n_records),
    'city': np.random.choice(['الرياض', 'جدة', 'مكة', 'المدينة', 'الدمام'], n_records),
    'product': np.random.choice(['لابتوب', 'هاتف', 'لوحي', 'سماعات', 'شاحن'], n_records),
    'category': np.random.choice(['إلكترونيات', 'أجهزة', 'ملحقات'], n_records),
    'quantity': np.random.randint(1, 5, n_records),
    'price': np.random.uniform(50, 2000, n_records).round(2),
    'cost': np.random.uniform(30, 1500, n_records).round(2),
}

df = pd.DataFrame(data)
df['profit'] = (df['price'] - df['cost']) * df['quantity']

print("📋 معلومات البيانات:")
print(df.info())

print("\n📊 أول 5 صفوف:")
print(df.head())

# 2. الإحصائيات الوصفية
print("\n" + "=" * 50)
print("2. الإحصائيات الوصفية")
print("=" * 50)

print("📊 الإحصائيات الوصفية للبيانات الرقمية:")
print(df.describe())

# 3. تحليل التوزيعات
print("\n" + "=" * 50)
print("3. تحليل التوزيعات")
print("=" * 50)

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"الأعمدة الرقمية: {list(numeric_cols)}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('توزيع البيانات الرقمية', fontsize=16, fontweight='bold')

for i, col in enumerate(numeric_cols[:4]):
    ax = axes[i // 2, i % 2]
    df[col].hist(bins=30, ax=ax, edgecolor='black')
    ax.set_title(f'توزيع {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('التكرار')

plt.tight_layout()
plt.show()  # إضافة plt.show() لعرض الرسوم

# 4. تحليل العلاقات والارتباطات
print("\n" + "=" * 50)
print("4. تحليل العلاقات والارتباطات")
print("=" * 50)

correlation_matrix = df[numeric_cols].corr()
print("📈 مصفوفة الارتباط:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', linewidths=0.5)
plt.title('مصفوفة الارتباط بين المتغيرات الرقمية', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()  # إضافة plt.show()

# 5. تحليل البيانات الفئوية
print("\n" + "=" * 50)
print("5. تحليل البيانات الفئوية")
print("=" * 50)

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"الأعمدة الفئوية: {list(categorical_cols)}")

for col in categorical_cols:
    print(f"\n📊 تحليل العمود: {col}")
    print(f"عدد القيم الفريدة: {df[col].nunique()}")
    if df[col].nunique() < 10:  # عرض القيم فقط إذا كانت محدودة
        print(f"القيم الأكثر تكراراً:")
        print(df[col].value_counts().head())

# 6. تصور البيانات الفئوية
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

product_counts = df['product'].value_counts().head(10)
axes[0, 0].barh(product_counts.index, product_counts.values)
axes[0, 0].set_title('أفضل 10 منتجات مبيعاً')
axes[0, 0].set_xlabel('عدد المبيعات')

city_sales = df.groupby('city')['price'].sum().sort_values(ascending=False).head(10)
axes[0, 1].barh(city_sales.index, city_sales.values)
axes[0, 1].set_title('إجمالي المبيعات حسب المدينة')
axes[0, 1].set_xlabel('إجمالي المبيعات')

category_sales = df.groupby('category')['price'].sum()
axes[1, 0].pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
axes[1, 0].set_title('توزيع المبيعات حسب الفئة')

product_profit = df.groupby('product')['profit'].mean().sort_values(ascending=False).head(8)
axes[1, 1].bar(product_profit.index, product_profit.values)
axes[1, 1].set_title('متوسط الربح حسب المنتج (أعلى 8)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()  # إضافة plt.show()

print("\n✅ تم الانتهاء من التحليل الاستكشافي بنجاح!")