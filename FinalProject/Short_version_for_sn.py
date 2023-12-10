#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import subprocess

# Install requirements from requirements.txt
subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True, shell=True)

# The rest of your Python script goes here

#!pip install -r requirements.txt || true


# In[ ]:


import pandas as pd
import requests as rq
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import math
import datetime as dt
import plotly.express as px
from tqdm import tqdm
import statistics
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# # Парсинг данных позиций

# Сначала мы создаём массив, который содержить все даты, за которые мы хотим получить данные

# In[ ]:


start_date = datetime(2020, 5, 8)
end_date = datetime.now()

current_date = start_date
array_date = []
while current_date <= end_date:
    array_date += [current_date.strftime("%Y-%m-%d")]
    current_date += timedelta(days=1)


# Тут мы формируем массив из дней, а затем, если сервер работает хорошо, то возвращаем данные за весь промежуток от 5 августа 2020 до сегодняшнего дня. 

# Если сервер перестаёт отвечать, тогда полученные данные мы собираем в файл и при повторном запуске программа проверить наши файлы и начнёт с последнего записанного дня.

# В качестве отладки печатается дата дня, который мы сейчас получаем. 

# In[ ]:


def add_new_day(date_today: str, ticker_name: str):
    date_yesterday = (datetime.strptime(date_today, '%Y-%m-%d') - timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://iss.moex.com/iss/analyticalproducts/futoi/securities/{ticker_name}.json?from={date_today}&till={date_today}&table_type=full"

    while True:
        try:
            # Ваш код, который может вызвать ошибку
            response = rq.get(url)
            # break  # Если ошибки не возникло, выходим из цикла
        except Exception as e:
            # Обработка других исключений
            print(f"Произошла ошибка: {e}")
        else:
            break

    # Проверяем статус ответа
    if response.status_code != 200:
        print("Ошибка при получении данных")
        add_new_day(date_today)

    df_for_add = pd.DataFrame(response.json()["futoi"]["data"], columns=response.json()["futoi"]['columns'])
    df_for_add.sort_values(by=['ticker', 'tradedate','tradetime']).reset_index().drop('index', axis=1)
   
    return df_for_add

ticker_names=['sn']

for ticker_name in ticker_names:
    if Path(f"{ticker_name}_full_date.csv").exists():
        df = pd.read_csv(f"{ticker_name}_full_date.csv")
        last_valid_date_index = df['tradedate'].last_valid_index()
        last_valid_date_value  = df['tradedate'][last_valid_date_index]
        date_tomorrow = (datetime.strptime(last_valid_date_value, '%Y-%m-%d') + timedelta(days=1)).strftime("%Y-%m-%d")
        print(date_tomorrow)
        if(not date_tomorrow in array_date):
            pass
        else:
            df_array = [df]
            for date in array_date[array_date.index(date_tomorrow)+1:]:
                df_array += [add_new_day(date, ticker_name)]
                print(date)
            df = pd.concat(df_array).sort_values(by=['ticker', 'tradedate', 'tradetime']).reset_index().drop('index', axis=1)
            print(df)
            filename = f"{ticker_name}_full_date.csv"
            df.to_csv(filename, sep=',', index=False, encoding='utf-8')
    else:
        print(f"Файл '{ticker_name}_full_date.csv' не существует.")
        df_array = []
        for date in array_date:
            df_array += [add_new_day(date, ticker_name)]
            print(date)
        df = pd.concat(df_array).sort_values(by=['ticker', 'tradedate', 'tradetime']).reset_index().drop('index', axis=1)
        print(df)
        filename = f"{ticker_name}_full_date.csv"
        df.to_csv(filename, sep=',', index=False, encoding='utf-8')

    


# Удаление повторяющихся данных (в силу того, как МосБиржа возвращала данные (в выходные дни вместо пустого запроса она возвращает последнюю цену за последний рабочий день, то есть 14*20 одинаковых записей, в силу чего эта ячейка позволяла сэкономить, примерно, 300 МБ))

# In[ ]:


for ticker_name in ticker_names:
    your_file = f"{ticker_name}_full_date.csv" 
# Загрузка данных
    df = pd.read_csv(your_file)

# Удаление повторяющихся строк
    unique_rows = df.drop_duplicates()

# Сохранение результата
    unique_rows.to_csv(your_file, index=False)


# Итоговые данные не идеальны, поэтому мы их будем обрабатывать в других данных.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Парсинг данных цен

# Сначала мы создаём массив, который содержить все даты, за которые мы хотим получить данные

# In[ ]:


start_date = datetime(2020, 5, 8)
end_date = datetime.now()

current_date = start_date
array_date = []
while current_date <= end_date:
    array_date += [current_date.strftime("%Y-%m-%d")]
    current_date += timedelta(days=1)


# Тут мы формируем массив из дней, а затем, если сервер работает хорошо, то возвращаем данные за весь промежуток от 5 августа 2020 до сегодняшнего дня. 

# Если сервер перестаёт отвечать, тогда полученные данные мы собираем в файл и при повторном запуске программа проверить наши файлы и начнёт с последнего записанного дня.

# В качестве отладки печатается дата дня, который мы сейчас получаем. 

# In[ ]:


def add_new_day(date_today: str, ticker_name: str): 
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/sessions/2/securities/{ticker_name}.json?from={date_today}&till={date_today}&table_type=full" 
 
    while True: 
        try: 
            # Ваш код, который может вызвать ошибку 
            response = rq.get(url) 
            # break  # Если ошибки не возникло, выходим из цикла 
        except Exception as e: 
            # Обработка других исключений 
            print(f"Произошла ошибка: {e}") 
        else: 
            break 
 
    # Проверяем статус ответа 
    if response.status_code != 200: 
        print("Ошибка при получении данных") 
        add_new_day(date_today) 
 
    df_for_add = pd.DataFrame(response.json()["history"]["data"], columns=response.json()["history"]['columns']) 
    df_for_add.sort_values(by=['TRADEDATE']).reset_index().drop('index', axis=1) 
    
 
    return df_for_add 
 
def process_ticker(ticker_name): 
    if (Path(f"{ticker_name}_full_date_price.csv").exists() or Path(f"{ticker_dict[ticker_name]}_full_date_price.csv")).exists():
        print(f"Файл '{ticker_name}_full_date_price.csv' существует.") 
        df = pd.read_csv(f"{ticker_name}_full_date_price.csv") 
        last_valid_date_index = df['TRADEDATE'].last_valid_index() 
        last_valid_date_value  = df['TRADEDATE'][last_valid_date_index] 
        date_tomorrow = (datetime.strptime(last_valid_date_value, '%Y-%m-%d') + timedelta(days=1)).strftime("%Y-%m-%d") 
        print(date_tomorrow) 
    # Вывод первых нескольких строк DataFrame для проверки 
        if(not date_tomorrow in array_date): 
            pass 
        else: 
            df_array = [df] 
            for date in array_date[array_date.index(date_tomorrow)+1:]: 
                df_array += [add_new_day(date, ticker_name)] 
                print(date) 
            df = pd.concat(df_array).sort_values(by=['TRADEDATE']).reset_index().drop('index', axis=1) 
            print(df) 
            filename = f"{ticker_name}_full_date_price.csv" 
            df.to_csv(filename, sep=',', index=False, encoding='utf-8') 
    else: 
        print(f"Файл '{ticker_name}_full_date_price.csv' не существует.") 
        df_array = [] 
        for date in array_date: 
            df_array += [add_new_day(date, ticker_name)] 
            print(date) 
        df = pd.concat(df_array).sort_values(by=['TRADEDATE']).reset_index().drop('index', axis=1) 
        print(df) 
        filename = f"{ticker_name}_full_date_price.csv" 
        df.to_csv(filename, sep=',', index=False, encoding='utf-8') 
        
ticker_names = ['SNGS']
short_ticker_names = ['sn']

ticker_dict = dict(zip(ticker_names, short_ticker_names))

with ThreadPoolExecutor() as executor: 
    executor.map(process_ticker, ticker_names)


# Удаление повторяющихся данных (в силу того, как МосБиржа возвращала данные (в выходные дни вместо пустого запроса она возвращает последнюю цену за последний рабочий день, то есть 14*20 одинаковых записей, в силу чего эта ячейка позволяла сэкономить, примерно, 300 МБ))

# In[ ]:


def rename_files(ticker_names):
    # Путь к папке с файлами
    folder_path = Path('.')

    # Получаем список файлов в папке
    files = [f for f in folder_path.iterdir() if f.is_file()]

    for file_path in files:
        # Проверяем, что файл имеет нужное расширение и соответствует шаблону
        if file_path.suffix == '.csv' and any(ticker in file_path.name for ticker in ticker_names):
            # Извлекаем имя тикера из имени файла
            for ticker, short_ticker in zip(ticker_names, ['sr', 'gz', 'lk', 'vb', 'rn', 'mn', 'af', 'al', 'sn', 'yn', 'tt', 'nm', 'hy', 'me', 'fv', 'gk', 'mg', 'ml']):
                if ticker in file_path.name:
                    # Формируем новое имя файла
                    new_file_name = file_path.name.replace(f"{ticker}_full_date_price", f"{short_ticker}_full_date_price")
                    # Составляем полные пути к старому и новому файлу
                    new_file_path = folder_path / new_file_name
                    # Переименовываем файл
                    file_path.rename(new_file_path)
                    print(f'Файл {file_path.name} переименован в {new_file_name}')
                    break
 
rename_files(['SBER', 'GAZP', 'LKOH', 'VTBR', 'ROSN', 'MGNT', 'AFLT', 'ALRS', 'SNGS', 'YNDX', 'TATN', 'NLMK', 'HYDR', 'MOEX', 'FIVE', 'GMKN', 'MAGN'])


# In[ ]:


ticker_names=['sn']

for ticker_name in ticker_names:
    your_file = f"{ticker_name}_full_date_price.csv" 
# Загрузка данных
    df = pd.read_csv(your_file)

# Удаление повторяющихся строк
    unique_rows = df.drop_duplicates()

# Сохранение результата
    unique_rows.to_csv(your_file, index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Стратегия и функция полезности

# Вначале мы задаем вспомогательный класс, представляющий собой открытую позицию, то есть записывает информацию об открытой позиции: размер позиции, цена и булево значение шорт или лонг позиция.

# ## Класс стратегии

# In[ ]:


class MyPosition:
    
    def __init__(self, amount: float, price_current: float, short: bool) -> None:
        """
        Инициализация объекта MyPosition.

        :param amount: Количество активa.
        :param price_current: Текущая цена актива.
        :param acc_fees: Количесво накопленных комиссий.
        :param short: Флаг короткой позиции (True, если короткая; False, если длинная).
        """
        self._amount: float = amount
        self._price_current: float = price_current
        self._acc_fees: float = 0
        self._short: bool = short

    def update_state(self, price: float) -> None:
        """
        Обновление состояния позиции.

        :param price: Новая цена актива.
        """
        self._price_current = price
        if self._short:
            # Рассчитываем комиссию за перенос короткой позиции
            transfer_fee = abs(self._amount) * self._price_current * 0.00065
            self._acc_fees += transfer_fee

    def balance(self) -> float:
        """
        Вычисление баланса позиции.

        :return: Баланс позиции.
        """
        return self._amount * self._price_current


# Теперь про класс стратегии:
# ### Поля Класса Стратегии
# _position: текущая открытая позиция (если есть).
# _states: данные о состоянии рынка (временная метка, открытые позиции, цена).
# _equity: баланс пользователя.
# _margin_equity: сумма использованной маржи.
# ### Основная Функция
# Метод run выполняет основную стратегию. Происходит построение скользящей средней, установка диапазона, итерационное отслеживание состояния рынка, а также открытие и закрытие позиций при соответствующих условиях.
# ### Вспомогательные Функции
# calc_upper_and_lower: расчет верхней и нижней границ скользящей средней.
# open_short: открытие короткой позиции.
# open_long: открытие длинной позиции.
# close_short: закрытие короткой позиции.
# close_long: закрытие длинной позиции.
# ### Параметры
# RISK: параметр риска.
# 
# STD_COUNT_UP, STD_COUNT_DOWN: коэффициенты для расчета границ скользящей средней.
# 
# MA_COUNT: ширина окна для построения скользящей средней.
# ### Вывод
# Результат выполнения стратегии записывается в массив, содержащий информацию о времени, цене актива, балансе пользователя, типе позиции, границах скользящей средней и флагах покупки/продажи.

# In[ ]:


class Strategy:

    params = {'FEE': 0.0004}

    def __init__(self, states, start_equity):
        self._position = None
        self._states = states
        self._equity = start_equity
        self._margin_equity = 0

    def run(self, RISK, STD_COUNT_UP, STD_COUNT_DOWN, MA_COUNT):
        states_ma = self._states['pos'].rolling(window=MA_COUNT).mean()
        data = []

        for i in tqdm(range(len(self._states))):
            if i < 10 * MA_COUNT:
                continue
            elif (i == 10 * MA_COUNT):
                state = self._states.loc[i]
                pos_ma = states_ma[i]
                prev_pos_ma = states_ma[i - 1]
                (pos_ma_upper, pos_ma_lower) = self.calc_upper_and_lower(i, states_ma, STD_COUNT_UP, STD_COUNT_DOWN)
            else:
                state = self._states.loc[i]
                pos_ma = states_ma[i]
                prev_pos_ma = states_ma[i - 1]

            b = 0
            s = 0

            if self._position:
                if state['price'] != self._states['price'][i - 1]:
                    self._position.update_state(state['price'])

                #if prev_pos_ma > pos_ma_lower and pos_ma < pos_ma_lower and self._position._short:
                if prev_pos_ma < pos_ma_upper and pos_ma > pos_ma_upper:
                    pos_ma_upper, pos_ma_lower = self.calc_upper_and_lower(i, states_ma, STD_COUNT_UP, STD_COUNT_DOWN)
                    if self._position._short:
                        b = 1
                #if prev_pos_ma < pos_ma_upper and pos_ma > pos_ma_upper and not self._position._short:
                if prev_pos_ma > pos_ma_lower and pos_ma < pos_ma_lower:
                    pos_ma_upper, pos_ma_lower = self.calc_upper_and_lower(i, states_ma, STD_COUNT_UP, STD_COUNT_DOWN)
                    if not self._position._short:
                        s = 1

                data.append([
                    state['datetime'],
                    state['price'],
                    self._equity - self._position._acc_fees,
                    self._position._short,
                    pos_ma_upper,
                    pos_ma_lower,
                    pos_ma,
                    b,
                    s
                ])

                if b:
                    self.close_short() 
                if s:
                    self.close_long()
            else:
                #if prev_pos_ma < pos_ma_lower and pos_ma > pos_ma_lower:
                if prev_pos_ma < pos_ma_upper and pos_ma > pos_ma_upper:
                    pos_ma_upper, pos_ma_lower = self.calc_upper_and_lower(i, states_ma, STD_COUNT_UP, STD_COUNT_DOWN)
                    self.open_long(state, RISK)
                    b = 1
                #if prev_pos_ma > pos_ma_upper and pos_ma < pos_ma_upper:
                if prev_pos_ma > pos_ma_lower and pos_ma < pos_ma_lower:
                    pos_ma_upper, pos_ma_lower = self.calc_upper_and_lower(i, states_ma, STD_COUNT_UP, STD_COUNT_DOWN)
                    self.open_short(state, RISK)
                    s = 1

                data.append([
                    state['datetime'],
                    state['price'],
                    self._equity,
                    0,
                    pos_ma_upper,
                    pos_ma_lower,
                    pos_ma,
                    b,
                    s
                ])

        return pd.DataFrame(data, columns=['datetime', 'price', 'equity', 'short', 'pos_ma_upper', 'pos_ma_lower', 'pos_ma', 'buy', 'sell'])

    def calc_upper_and_lower(self, i, states_ma, STD_COUNT_UP, STD_COUNT_DOWN):
        #mean_pos_ma = states_ma.head(i).mean()
        std_pos_ma = states_ma.head(i).std()
        pos_ma_upper = states_ma[i] + STD_COUNT_UP * std_pos_ma
        pos_ma_lower = states_ma[i] - STD_COUNT_DOWN * std_pos_ma
        return pos_ma_upper, pos_ma_lower

    def open_short(self, state, RISK):
        if self._position:
            raise Exception(f'Cannot open position, already have one {self._position}')

        self._margin_equity += (np.floor((1 - self.params['FEE']) * RISK * self._equity / state['price']) * state['price'] - self.params['FEE'] * self._equity)

        amount = (-1) * np.floor((1 - self.params['FEE']) * RISK * self._equity / state['price'])

        self._position = MyPosition(amount, state['price'], True)

    def open_long(self, state, RISK):
        if self._position:
            raise Exception(f'Cannot open position, already have one {self._position}')

        self._margin_equity -= (np.floor((1 - self.params['FEE']) * RISK * self._equity / state['price']) * state['price'] + self.params['FEE'] * RISK * self._equity)

        amount = np.floor((1 - self.params['FEE']) * RISK * self._equity / state['price'])

        self._position = MyPosition(amount, state['price'], False)

    def close_short(self):
        if self._position._short == 0:
            raise Exception(f'Cannot close short position, it is long')

        self._equity += (
                self._margin_equity + (1 - self.params['FEE']) * self._position.balance() - self._position._acc_fees)
        self._margin_equity = 0
        self._position = None

    def close_long(self):
        if self._position._short == 1:
            raise Exception(f'Cannot close long position, it is short')

        self._equity += (self._margin_equity + (1 - self.params['FEE']) * self._position.balance() - self._position._acc_fees)
        self._margin_equity = 0
        self._position = None


# In[ ]:


def run_strategy(all_data_copy, std_count_up, std_count_down, ma_count):
    strategy = Strategy(all_data_copy, 10000000)
    df = pd.DataFrame(strategy.run(1, std_count_up, std_count_down, ma_count))
    df['equity'] = df['equity'] / df['equity'].iloc[0]
    final_total_balance = df['equity'].iloc[-1]
    return final_total_balance

# Тренировка

def train(all_data_copy1: pd.DataFrame) -> (float, float, float):
    # Грубый поиск параметров
    std_counts_up = np.arange(0.25, 1, 0.25)
    std_counts_down = np.arange(0.25, 1, 0.25)
    ma_counts = np.arange(500, 3000, 750)

    radius = 2

    parameters = []
    final_total_balances = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for std_count_up in std_counts_up:
            for std_count_down in std_counts_down:
                for ma_count in ma_counts:
                    #print(std_count_up, std_count_down, ma_count)
                    future = executor.submit(run_strategy, all_data_copy1, std_count_up, std_count_down, ma_count)
                    futures.append((std_count_up, std_count_down, ma_count, future))

        for std_count_up, std_count_down, ma_count, future in futures:
            final_total_balance = future.result()
            parameters.append((std_count_up, std_count_down, ma_count))
            final_total_balances.append(final_total_balance)

    data_parametrs = pd.DataFrame(parameters, columns=['std_count_up', 'std_count_down', 'ma_count'])
    data_parametrs['final_total_balance'] = final_total_balances

    array_utility = []
    for index_std_count_up in range(len(std_counts_up)):
        for index_std_count_down in range(len(std_counts_down)):
            for index_ma_count in range(len(ma_counts)):
                array_utility.append(Utility_assess(data_parametrs, index_std_count_up, index_std_count_down, index_ma_count, len(std_counts_up), len(std_counts_down), len(ma_counts), radius))

    max_utility = max(array_utility)
    max_utility_index = array_utility.index(max_utility)
    corresponding_parameters = parameters[max_utility_index] 

    # Уточнение параметров
    std_counts_up = np.arange(corresponding_parameters[0] - 0.1, corresponding_parameters[0] + 0.2, 0.1)
    std_counts_down = np.arange(corresponding_parameters[1] - 0.1, corresponding_parameters[1] + 0.2, 0.1)
    ma_counts = np.arange(corresponding_parameters[2] - 250, corresponding_parameters[2] + 500, 250)

    parameters = []
    final_total_balances = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for std_count_up in std_counts_up:
            for std_count_down in std_counts_down:
                for ma_count in ma_counts:
                    future = executor.submit(run_strategy, all_data_copy1, std_count_up, std_count_down, ma_count)
                    futures.append((std_count_up, std_count_down, ma_count, future))

        for std_count_up, std_count_down, ma_count, future in futures:
            final_total_balance = future.result()
            parameters.append((std_count_up, std_count_down, ma_count))
            final_total_balances.append(final_total_balance)

    data_parametrs = pd.DataFrame(parameters, columns=['std_count_up', 'std_count_down', 'ma_count'])
    data_parametrs['final_total_balance'] = final_total_balances

    array_utility = []
    for index_std_count_up in range(len(std_counts_up)):
        for index_std_count_down in range(len(std_counts_down)):
            for index_ma_count in range(len(ma_counts)):
                array_utility.append(Utility_assess(data_parametrs, index_std_count_up, index_std_count_down, index_ma_count, len(std_counts_up), len(std_counts_down), len(ma_counts), radius))

    max_utility = max(array_utility)
    max_utility_index = array_utility.index(max_utility)
    corresponding_parameters = parameters[max_utility_index] 
    corresponding_final_total_balance = final_total_balances[max_utility_index]   

    print(f"Maximum Total Balance: {corresponding_final_total_balance}")
    print(f"Maximum Utility: {corresponding_final_total_balance}")
    print(f"Corresponding Parameters (std_count, std_timerange): {corresponding_parameters}")
    return corresponding_parameters

# Тест

def test(all_data_copy2: pd.DataFrame, std_count_up: float, std_count_down: float, ma_count: float) -> float:
    strategy = Strategy(all_data_copy2, 10000000)
    df = pd.DataFrame(strategy.run(1, std_count_up, std_count_down, ma_count))
    df['equity'] = df['equity'] / df['equity'].iloc[0]
    final_total_balance = df['equity'].iloc[-1]

    print(f'Test profit = {final_total_balance}')
    return final_total_balance


# Очистка данных от повторяющихся строчек (возникают в силу того, как МосБиржа возвращает данные при запросах данных выходных дней)

# In[ ]:


def clear_data(tiker):
    # Pos

    df = pd.read_csv(f'{tiker}_full_date.csv', sep=",")
    df = df.sort_values(by=['ticker', 'tradedate', 'tradetime']).drop_duplicates().reset_index().drop('index', axis=1)
    df['datetime'] = pd.to_datetime(df['tradedate'] + ' ' + df['tradetime']) 
    df = df.drop(['tradedate', 'tradetime'], axis=1) 
    df = df[df['clgroup'] == 'YUR'].reset_index().drop('index', axis=1)
    df = df[['datetime', 'pos']]
    data_pos = df

    # Price

    df = pd.read_csv(f'{tiker}_full_date_price.csv', sep=",")
    df.reset_index(inplace=True)
    df.rename(columns={'TRADEDATE': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['BOARDID']=='TQBR']
    df = df[['datetime', 'WAPRICE']]
    df.rename(columns={'WAPRICE': 'price'}, inplace=True)
    data_price = df

    # All data

    price_dict = dict(zip(data_price['datetime'].dt.date, data_price['price']))

    data_pos['price'] = data_pos['datetime'].dt.date.map(price_dict)

    all_data = data_pos.sort_values('datetime').drop_duplicates().dropna().reset_index().drop('index', axis=1)

    all_data_copy = all_data.copy()

    all_data_copy['date'] = all_data_copy['datetime'].dt.date

    # For train

    train_data = all_data_copy[all_data_copy['date'] < pd.to_datetime('2022-02-20').date()].reset_index().drop('index', axis=1)

    # For test

    test_data = all_data_copy[all_data_copy['date'] > pd.to_datetime('2022-12-31').date()].reset_index().drop('index', axis=1)

    return train_data, test_data


# ## Функция полезности
# 
# Тут мы задаём некоторую функцию полезности и затем считаем ее для каждого параметра, в качестве радиуса мы берём 2, то есть по рассматриваем дисперсию окресности из 125 $((2*2+1)^3)$ точек. 

# In[ ]:


def Utility_fun(income, var):
    return income - 10*var

def Utility_assess(df_plotly, index_std_count_up, index_std_count_down, index_ma_count, len_std_counts_up, len_std_counts_down, len_ma_counts, radius):
    # Используем iloc для явного указания целочисленного индекса
    income = df_plotly.loc[
        (df_plotly['std_count_up'] == df_plotly['std_count_up'].iloc[index_std_count_up]) &
        (df_plotly['std_count_down'] == df_plotly['std_count_down'].iloc[index_std_count_down]) &
        (df_plotly['ma_count'] == df_plotly['ma_count'].iloc[index_ma_count])
    ]['final_total_balance'].values[0]

    index_std_count_up -= radius
    index_std_count_down -= radius
    index_ma_count -= radius

    array_income = np.array([])
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            for k in range(2 * radius + 1):
                if (
                    (index_std_count_up + i >= 0) and
                    (index_std_count_down + j >= 0) and
                    (index_ma_count + k >= 0) and
                    (index_std_count_up + i < len_std_counts_up) and
                    (index_std_count_down + j < len_std_counts_down) and
                    (index_ma_count + k < len_ma_counts)
                ):
                    array_income = np.append(
                        array_income,
                        df_plotly.loc[
                            (df_plotly['std_count_up'] == df_plotly['std_count_up'].iloc[index_std_count_up + i]) &
                            (df_plotly['std_count_down'] == df_plotly['std_count_down'].iloc[index_std_count_down + j]) &
                            (df_plotly['ma_count'] == df_plotly['ma_count'].iloc[index_ma_count + k])
                        ]['final_total_balance'].values[0]
                    )

    std_of_income = statistics.stdev(array_income) if len(array_income) >= 2 else 0
    Utility = Utility_fun(income, std_of_income)
    return Utility


# Требуется подождать 5-7 минут

# In[ ]:


ticker_names=['sn']

for ticker in ticker_names:

    train_data, test_data = clear_data(ticker)

    corresponding_parameters = []

    corresponding_parameters = train(train_data)
    result = test(test_data, corresponding_parameters[0], corresponding_parameters[1], corresponding_parameters[2])


# In[ ]:


train_data, test_data = clear_data('sn')

strategy = Strategy(train_data, 10000000)
df = pd.DataFrame(strategy.run(1, 1, 1, 500))

px.line(df, x='datetime', y=['pos_ma', 'pos_ma_upper', 'pos_ma_lower']).show()

px.line(df, x='datetime', y='equity').update_xaxes(type='category').show()

up = pd.DataFrame(columns=['datetime', 'price'])
down = pd.DataFrame(columns=['datetime', 'price'])
for i in range(len(df['price'])):
    if df['buy'][i]:
        up.loc[len(up)] = df.iloc[i]
    elif df['sell'][i]:
        down.loc[len(down)] = df.iloc[i]
fig = px.line(df, x='datetime', y='price', title='График цен по времени', labels={'datetime': 'Дата и время', 'price': 'Цена'})
fig.add_trace(px.scatter(up, x='datetime', y='price').update_traces(marker=dict(color='red')).data[0])
fig.add_trace(px.scatter(down, x='datetime', y='price').update_traces(marker=dict(color='green')).data[0])
fig.update_xaxes(type='category')
fig.show()


# ## Функция полезности
# 
# Тут мы задаём некоторую функцию полезности и затем считаем ее для каждого параметра, в качестве радиуса мы берём 2, то есть по рассматриваем дисперсию окресности из 125 $((2*2+1)^3)$ точек. 

# In[ ]:


def Utility_fun(income, var):
    return income - 10*var


# In[ ]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots

step_std_counts_up = 0.2
step_std_counts_down = 0.2
step_ma_counts = 300
std_counts_up = np.arange(0.2, 1.01, step_std_counts_up)
std_counts_down = np.arange(0.2, 1.01, step_std_counts_down)
ma_counts = np.arange(500, 2000, step_ma_counts)

parameters = []
final_total_balances = []

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    
    for std_count_up in std_counts_up:
        for std_count_down in std_counts_down:
            for ma_count in ma_counts:
                future = executor.submit(run_strategy, train_data, std_count_up, std_count_down, ma_count)
                futures.append((std_count_up, std_count_down, ma_count, future))

    for std_count_up, std_count_down, ma_count, future in futures:
        final_total_balance = future.result()
        parameters.append((std_count_up, std_count_down, ma_count))
        final_total_balances.append(final_total_balance)

# Преобразование в DataFrame для удобства работы с данными
df_plotly = pd.DataFrame(parameters, columns=['std_count_up', 'std_count_down', 'ma_count'])
df_plotly['final_total_balance'] = final_total_balances
print(df_plotly)


# Создание трехмерного графика в Plotly
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

scatter = go.Scatter3d(
    x=df_plotly['std_count_up'],
    y=df_plotly['std_count_down'],
    z=df_plotly['ma_count'],
    mode='markers',
    marker=dict(
        size=8,
        color=df_plotly['final_total_balance'],
        colorscale='Viridis',
        colorbar=dict(title='Final Total Balance')
    )
)

fig.add_trace(scatter)

# Наименование осей
fig.update_layout(scene=dict(
                    xaxis_title='std_count_up',
                    yaxis_title='std_count_down',
                    zaxis_title='ma_count')
                 )

# Отображение графика
fig.show()


# In[ ]:


radius = 2

array_utility = []
for index_std_count_up in range(len(std_counts_up)):
    for index_std_count_down in range(len(std_counts_down)):
        for index_ma_count in range(len(ma_counts)):
            array_utility.append(Utility_assess(df_plotly, index_std_count_up, index_std_count_down, index_ma_count, len(std_counts_up), len(std_counts_down), len(ma_counts), radius))

df_plotly['final_utility_balance'] = array_utility
# print(np.max(df_plotly['final_total_balance']), np.max(array_utility))


# In[ ]:


# Создание трехмерного графика в Plotly
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

scatter = go.Scatter3d(
    x=df_plotly['std_count_up'],
    y=df_plotly['std_count_down'],
    z=df_plotly['ma_count'],
    mode='markers',
    marker=dict(
        size=4,
        color=df_plotly['final_utility_balance'],
        colorscale='Viridis',
        colorbar=dict(title='Final Total Balance')
    ),
    text=df_plotly.apply(lambda row: f' final_utility_balance: {row["final_utility_balance"]},  final_total_balance: {row["final_total_balance"]}', axis=1)
)

fig.add_trace(scatter)

# Наименование осей
fig.update_layout(scene=dict(
                    xaxis_title='std_count_up',
                    yaxis_title='std_count_down',
                    zaxis_title='ma_count')
                 )

# Отображение графика
fig.show()

