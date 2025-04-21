import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task1():
    # Загрузка данных
    tr_mcc_codes = pd.read_csv('D:\\Downloads\\data\\tr_mcc_codes.csv', sep=';')
    transactions = pd.read_csv('D:\\Downloads\\data\\transactions.csv', sep=',', nrows=1000000)
    customer_gender_train = pd.read_csv('D:\\Downloads\\data\\gender_train.csv', sep=',')
    tr_types = pd.read_csv('D:\\Downloads\\data\\tr_types.csv', sep=';')
    # Объединение данных
    merged = transactions.merge(tr_types, on='tr_type', how='inner').\
        merge(tr_mcc_codes, on='mcc_code', how='inner')
    # Объеднение через left join
    merged = merged.merge(customer_gender_train, on='customer_id', how='left')
    merged = merged[merged['amount'] < 0]
    merged['tr_hour'] = merged['tr_datetime'].apply(lambda x: x.split(':')[0][-2:])
    bins = [-float('inf'), -10000, -5000, -1000, -500, -100, 0]
    labels = ['<-10k', '-10k to -5k', '-5k to -1k', '-1k to -500', '-500 to -100', '-100 to 0']
    merged['amount_bucket'] = pd.cut(merged['amount'], bins=bins, labels=labels)

    # Выбираем ночные часы (1-5)
    night_hours = ['01', '02', '03', '04', '05']
    night_trans = merged[merged['tr_hour'].isin(night_hours)]
    # Считаем распределение по полу
    gender_dist = night_trans['gender'].value_counts(normalize=True)
    # Проверяем условие
    female_ratio = gender_dist.get(1, 0)
    result = female_ratio > 0.85
    
    print(f"Доля женских транзакций ночью (1-5 ч): {female_ratio:.2%}")
    print(f"Предположение, что 85% ночных поступлений являются женскими: {'подтверждается' if result else 'не подтверждается'}")

    # Выбираем транзакции в 3 часа ночи
    three_am = merged[merged['tr_hour'] == '03']
    # Находим 10% самых низких трат
    threshold = three_am['amount'].quantile(0.1)
    lowest_spends = three_am[three_am['amount'] <= threshold]
    # Считаем распределение по полу
    gender_dist = lowest_spends['gender'].value_counts(normalize=True)
    # Проверяем условие
    female_ratio = gender_dist.get(1, 0)
    result = female_ratio > 0.7
    
    print(f"Доля женских транзакций среди самых низких трат в 3 часа: {female_ratio:.2%}")
    print(f"Предположение, что самые низкие траты в 70% случаев являются женскими: {'подтверждается' if result else 'не подтверждается'}")

    # Строим сводную таблицу
    pivot_table = pd.pivot_table(merged, 
                            values='gender', 
                            index='tr_hour', 
                            columns='amount_bucket', 
                            aggfunc='mean',
                            fill_value=0, observed=True
                            )
    return pivot_table

def plot_pivot_table(pivot_table):
    plt.figure(figsize=(9, 11))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='g', annot_kws={'fontsize': 14})
    plt.xticks(fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.xlabel('Bucket', size=18)
    plt.ylabel('Hours', fontsize=18)
    plt.title('Gender analysis per bucket and hour', fontsize=20)
    plt.show()