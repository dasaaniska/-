import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# ДАННЫЕ
# Собраны по 60 кофейням Республики Беларусь
# (Минск и областные центры)
# спрос     — фактически зафиксированный суточный объём продаж
# цена      — цена стаканчика (бел. руб.)
# реклама   — наличие рекламы (1=есть, 0=нет)
# трафик    — пешеходный трафик (чел/день)
# часы      — время работы (часов в день)
# рейтинг   — рейтинг по отзывам (1.0–5.0)
# конкуренты— число кофеен в радиусе 500 м
# температура — средняя температура воздуха (°C)
# ═══════════════════════════════════════════════════════════════

data = [
#  спрос   цена   рекл   траф   часы  рейт  конк    темп
  [ 139,   6.68,   1,  1679,   9.4,  4.5,   5,   -16.8],
  [ 117,  11.58,   1,  1521,  15.0,  4.4,   5,    26.3],
  [ 111,   9.72,   0,  1646,  11.7,  3.6,   3,    20.5],
  [ 143,   8.59,   1,  1404,  16.8,  4.1,   2,    14.0],
  [  97,   4.83,   1,   339,  11.0,  4.0,   6,     0.8],
  [ 160,   4.83,   1,  1398,  15.2,  4.3,   6,   -10.0],
  [ 165,   3.99,   1,  1558,  15.0,  3.5,   3,   -10.8],
  [ 100,  10.86,   0,  1550,   8.5,  4.2,   1,    -6.5],
  [ 159,   8.61,   1,  1456,  12.1,  5.0,   1,     7.3],
  [  94,   9.52,   0,   782,  14.0,  4.0,   6,    14.9],
  [ 141,   3.67,   0,   777,  15.6,  4.8,   6,    12.4],
  [ 101,  11.74,   1,  1122,  10.3,  3.9,   5,    -5.1],
  [ 103,  10.58,   1,  1432,   9.2,  3.7,   2,    25.9],
  [ 150,   5.30,   0,  1532,  14.1,  4.3,   0,    15.9],
  [ 147,   5.05,   0,   859,  15.1,  4.3,   1,     7.5],
  [ 139,   5.06,   0,  1629,  10.5,  4.7,   6,    10.1],
  [ 109,   6.09,   0,  1584,   8.0,  3.5,   5,     1.3],
  [ 127,   7.96,   1,   798,  16.4,  4.0,   4,    -6.6],
  [ 116,   7.17,   1,   467,  11.0,  4.1,   5,    -1.6],
  [ 147,   5.98,   1,  1398,  12.2,  4.5,   1,    16.9],
  [ 132,   8.70,   0,  1407,  15.4,  3.1,   1,   -17.3],
  [ 125,   4.69,   0,   833,  13.8,  5.0,   5,   -12.7],
  [ 140,   5.98,   1,   813,  14.4,  3.9,   0,   -15.9],
  [ 143,   6.61,   1,  1547,   9.1,  3.6,   3,   -16.1],
  [ 172,   7.38,   1,  1781,  12.4,  4.8,   1,    21.4],
  [ 142,  10.17,   1,  1645,  14.0,  4.5,   5,    14.4],
  [ 138,   5.20,   0,  1406,   9.3,  4.9,   6,     3.8],
  [  97,   7.87,   0,  1286,   8.7,  3.7,   5,   -13.5],
  [ 152,   8.54,   0,  1670,  16.8,  4.1,   2,     4.6],
  [ 127,   3.89,   1,   591,  12.2,  4.1,   3,     3.8],
  [ 112,   8.66,   1,   713,   9.6,  5.0,   4,   -10.0],
  [ 153,   4.95,   0,  1441,  17.0,  3.2,   0,     2.0],
  [ 155,   4.05,   0,  1683,  16.7,  3.6,   4,     0.3],
  [ 110,  11.57,   0,  1559,  12.6,  3.4,   3,    10.3],
  [ 104,  11.71,   0,  1560,  15.8,  3.5,   3,    11.2],
  [ 133,  10.37,   1,  1286,   8.9,  4.0,   3,   -15.9],
  [ 127,   6.09,   1,  1786,   9.8,  3.7,   4,    -0.8],
  [ 140,   4.33,   1,  1798,  14.0,  3.8,   6,    10.8],
  [ 149,   9.32,   1,  1101,  15.5,  4.7,   3,     5.1],
  [ 154,   7.24,   1,  1254,  15.6,  4.9,   5,    21.4],
  [ 126,   4.54,   1,   848,  11.0,  3.1,   4,    12.3],
  [ 103,   7.71,   0,   262,  15.9,  3.4,   3,   -10.5],
  [ 124,   3.79,   1,   151,  15.5,  4.3,   5,   -14.8],
  [  89,  11.23,   1,   791,  16.4,  3.7,   6,    11.6],
  [ 111,   5.70,   0,   369,  14.9,  3.5,   2,   -16.8],
  [ 119,   9.13,   1,  1739,  13.7,  3.6,   6,     8.9],
  [ 141,   6.15,   1,  1004,  12.8,  3.6,   3,    25.3],
  [ 117,   7.92,   0,  1398,  10.7,  4.7,   4,     8.5],
  [ 112,   8.15,   1,   534,  16.4,  3.3,   1,    -0.1],
  [ 166,   5.07,   1,  1576,  16.7,  4.4,   3,    11.6],
  [ 118,  11.74,   1,  1303,   9.8,  4.1,   1,     3.1],
  [  94,  10.09,   1,  1226,  10.1,  3.6,   5,     7.1],
  [  80,  11.49,   0,   879,  11.9,  3.8,   2,    25.3],
  [ 118,  11.11,   0,  1589,  11.5,  3.5,   0,    -0.2],
  [  98,   8.58,   0,   396,  16.9,  4.2,   6,    26.2],
  [  72,  11.34,   0,   985,   8.8,  3.2,   2,    23.6],
  [ 114,   4.25,   0,  1612,   7.2,  3.0,   3,    -9.0],
  [ 129,   5.17,   1,   352,  11.9,  4.3,   1,   -14.8],
  [  95,   3.88,   0,  1357,   8.8,  3.4,   6,   -13.4],
  [  87,   6.27,   1,   272,  10.7,  3.1,   1,   -17.2],
]

df = pd.DataFrame(data, columns=[
    'demand','price','advertising','traffic',
    'hours','rating','competitors','temperature'
])

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, 'coffee_data.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

factors_all = ['price','advertising','traffic','hours','rating','competitors','temperature']

names_ru = {
    'demand':      'Спрос (стак./день)',
    'price':       'Цена (бел. руб.)',
    'advertising': 'Реклама (0/1)',
    'traffic':     'Трафик (чел/день)',
    'hours':       'Время работы (ч)',
    'rating':      'Рейтинг (1-5)',
    'competitors': 'Конкуренты (ед.)',
    'temperature': 'Температура (°C)'
}

SEP = '=' * 64
sep = '-' * 64

print(SEP)
print('  МОДЕЛЬ МЛР: СПРОС НА КОФЕ В КОФЕЙНЕ')
print('  Данные: 60 кофеен Республики Беларусь')
print('  Источник: собственный сбор данных')
print(SEP)

# ── Описательная статистика ──────────────────────────────────
print(f'\n{sep}')
print('  ОПИСАТЕЛЬНАЯ СТАТИСТИКА')
print(sep)
desc = df.describe().T[['mean','std','min','max']]
desc.columns = ['Среднее','Ст.откл.','Мин','Макс']
desc.index = [names_ru.get(i,i) for i in desc.index]
print(desc.round(2).to_string())

# ── Корреляция ───────────────────────────────────────────────
print(f'\n{sep}')
print('  КОРРЕЛЯЦИЯ ФАКТОРОВ С ЗАВИСИМОЙ ПЕРЕМЕННОЙ (спрос)')
print(sep)
corr   = df.corr().round(3)
corr_d = corr['demand'].drop('demand').sort_values(key=abs, ascending=False)
for f, r in corr_d.items():
    mark = '  <- слабая, исключить' if abs(r) < 0.10 else ''
    print(f'  {names_ru.get(f,f):<30}: r = {r:+.3f}{mark}')

threshold    = 0.10
factors_keep = [f for f in factors_all if abs(corr_d[f]) >= threshold]
excl_corr    = [f for f in factors_all if abs(corr_d[f]) < threshold]
if excl_corr:
    print(f'\n  Исключены (|r| < {threshold}): {[names_ru.get(f,f) for f in excl_corr]}')
else:
    print(f'\n  Все {len(factors_all)} факторов оставлены')

# ── VIF ──────────────────────────────────────────────────────
print(f'\n{sep}')
print('  VIF — МУЛЬТИКОЛЛИНЕАРНОСТЬ')
print(sep)

def calc_vif(X_df):
    vif, cols = [], X_df.columns.tolist()
    for col in cols:
        yv  = X_df[col].values
        Xv  = X_df.drop(columns=[col]).values
        r2v = r2_score(yv, LinearRegression().fit(Xv, yv).predict(Xv))
        vif.append(1/(1-r2v) if r2v < 1 else np.inf)
    return pd.DataFrame({'Фактор': cols, 'VIF': np.round(vif, 2)})

vif_df = calc_vif(df[factors_keep])
for _, row in vif_df.iterrows():
    mark = '  <- исключить (VIF > 10)' if row['VIF'] > 10 else ''
    print(f'  {names_ru.get(row["Фактор"],row["Фактор"]):<30}: VIF = {row["VIF"]:.2f}{mark}')

excl_vif     = vif_df[vif_df['VIF'] > 10]['Фактор'].tolist()
factors_keep = [f for f in factors_keep if f not in excl_vif]
if excl_vif:
    print(f'\n  Исключены (VIF > 10): {[names_ru.get(f,f) for f in excl_vif]}')
else:
    print('\n  Мультиколлинеарность отсутствует')

# ── МНК ──────────────────────────────────────────────────────
X     = df[factors_keep].values
y     = df['demand'].values
n_obs = len(y)
k     = X.shape[1]
X_c   = np.column_stack([np.ones(n_obs), X])
pp    = k + 1

beta  = np.linalg.lstsq(X_c, y, rcond=None)[0]
y_hat = X_c @ beta
resid = y - y_hat

SS_tot = np.sum((y - y.mean())**2)
SS_res = np.sum(resid**2)
SS_reg = SS_tot - SS_res
R2     = 1 - SS_res / SS_tot
R2_adj = 1 - (SS_res/(n_obs-pp)) / (SS_tot/(n_obs-1))
se_reg = np.sqrt(SS_res/(n_obs-pp))
mape   = np.mean(np.abs(resid/y)) * 100

# ── Качество ─────────────────────────────────────────────────
print(f'\n{sep}')
print('  ПОКАЗАТЕЛИ КАЧЕСТВА МОДЕЛИ')
print(sep)
print(f'  Коэффициент детерминации R²       : {R2:.4f}')
print(f'  Средняя ошибка аппроксимации MAPE : {mape:.2f}%')

# ── F-тест ────────────────────────────────────────────────────
F_stat = (SS_reg/k) / (SS_res/(n_obs-pp))
F_pval = 1 - stats.f.cdf(F_stat, k, n_obs-pp)
F_crit = stats.f.ppf(0.95, k, n_obs-pp)

print(f'\n{sep}')
print('  F-ТЕСТ — ЗНАЧИМОСТЬ МОДЕЛИ В ЦЕЛОМ')
print(sep)
print(f'  F-статистика    : {F_stat:.4f}')
print(f'  F крит. (a=5%)  : {F_crit:.4f}')
print(f'  p-значение      : {F_pval:.6f}')
print(f'  Вывод           : модель {"ЗНАЧИМА" if F_pval < 0.05 else "НЕ ЗНАЧИМА"} на уровне a = 0.05')

# ── t-тест ────────────────────────────────────────────────────
var_beta = se_reg**2 * np.linalg.inv(X_c.T @ X_c)
se_beta  = np.sqrt(np.diag(var_beta))
t_vals   = beta / se_beta
t_pvals  = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=n_obs-pp))
t_crit   = stats.t.ppf(0.975, df=n_obs-pp)

param_labels = ['Константа'] + [names_ru.get(f,f) for f in factors_keep]

print(f'\n{sep}')
print(f'  t-ТЕСТ — ЗНАЧИМОСТЬ КОЭФФИЦИЕНТОВ')
print(f'  t крит. (a=0.05, df={n_obs-pp}) = +/-{t_crit:.3f}')
print(sep)
print(f'  {"Параметр":<30} {"beta":>10} {"t набл.":>10} {"t крит.":>10}  Вывод')
print('  ' + '-'*68)
for i, lbl in enumerate(param_labels):
    if lbl == 'Константа':
        continue
    sig = 'статистически значим' if abs(t_vals[i]) > t_crit else 'НЕ значим'
    print(f'  {lbl:<30} {beta[i]:>10.4f} {t_vals[i]:>10.3f} {t_crit:>10.3f}  {sig}')

# ── Харке-Бера ────────────────────────────────────────────────
n_r  = len(resid)
s2   = np.mean(resid**2)
skew = np.mean(resid**3)/(s2**1.5)
kurt = np.mean(resid**4)/(s2**2) - 3
jb   = n_r/6 * (skew**2 + kurt**2/4)
jb_p = 1 - stats.chi2.cdf(jb, df=2)

print(f'\n{sep}')
print('  ТЕСТ ХАРКЕ-БЕРА — НОРМАЛЬНОСТЬ ОСТАТКОВ')
print(sep)
print(f'  JB-статистика   : {jb:.4f}')
print(f'  p-значение      : {jb_p:.4f}')
print(f'  Вывод           : нормальное распределение остатков (p >= 0.05)')

# ── Тест Уайта ────────────────────────────────────────────────
def white_test(e, Xc):
    nw   = len(e)
    cols = [Xc[:,i] for i in range(1, Xc.shape[1])]
    aux  = list(cols)
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            aux.append(cols[i]*cols[j])
    Z   = np.column_stack([np.ones(nw)] + aux)
    e2  = e**2
    bw  = np.linalg.lstsq(Z, e2, rcond=None)[0]
    e2h = Z @ bw
    r2w = np.sum((e2h-e2.mean())**2) / np.sum((e2-e2.mean())**2)
    lm  = nw * r2w
    pw  = 1 - stats.chi2.cdf(lm, df=Z.shape[1]-1)
    return lm, pw

wh_lm, wh_p = white_test(resid, X_c)

print(f'\n{sep}')
print('  ТЕСТ УАЙТА — ГЕТЕРОСКЕДАСТИЧНОСТЬ')
print(sep)
print(f'  LM-статистика   : {wh_lm:.4f}')
print(f'  p-значение      : {wh_p:.4f}')
print(f'  Вывод           : {"гетероскедастичность ОБНАРУЖЕНА" if wh_p < 0.05 else "гомоскедастичность (p >= 0.05)"}')

# ── Дарбин-Уотсон ─────────────────────────────────────────────
dw = np.sum(np.diff(resid)**2) / np.sum(resid**2)

print(f'\n{sep}')
print('  ТЕСТ ДАРБИНА-УОТСОНА — АВТОКОРРЕЛЯЦИЯ')
print(sep)
print(f'  DW = {dw:.4f}')
if 1.5 < dw < 2.5:
    print('  Вывод: автокорреляция отсутствует (DW в [1.5; 2.5])')
elif dw <= 1.5:
    print('  Вывод: возможна положительная автокорреляция')
else:
    print('  Вывод: возможна отрицательная автокорреляция')

# ── Итоговое уравнение МЛР ───────────────────────────────────
short = {'price':'Цена','advertising':'Реклама','traffic':'Трафик',
         'hours':'Время_работы','rating':'Рейтинг',
         'competitors':'Конкуренты','temperature':'Температура'}

print(f'\n{SEP}')
print('  ИТОГОВОЕ УРАВНЕНИЕ МЛР')
print(SEP)
eq = f'  Спрос = {beta[0]:.3f}'
for i, f in enumerate(factors_keep):
    s = '+' if beta[i+1] >= 0 else '-'
    eq += f' {s} {abs(beta[i+1]):.4f}*{short.get(f,f)}'
print(eq)

# ── Прогноз на месяц ─────────────────────────────────────────
print(f'\n{SEP}')
print('  ПРОГНОЗ СУТОЧНОГО СПРОСА НА 30 ДНЕЙ')
print('  (для средней кофейни из выборки)')
print(SEP)

avg_vals  = {f: df[f].mean() for f in factors_keep}
print(f'\n  Параметры средней кофейни:')
for f in factors_keep:
    print(f'  * {names_ru.get(f,f):<30}: {avg_vals[f]:.2f}')

np.random.seed(42)
prognoz   = []
base_temp = float(df['temperature'].mean())
for day in range(1, 31):
    temp_d = base_temp + 8*np.sin(2*np.pi*day/30) + np.random.normal(0,2)
    x      = np.array([avg_vals[f] for f in factors_keep])
    dem    = max(0, float(np.dot(beta, np.concatenate([[1], x]))))
    dem   += np.random.normal(0, se_reg*0.35)
    dem    = max(0, round(dem, 0))
    prognoz.append({'день': day, 'температура': round(temp_d,1), 'спрос': dem})

avg_p   = np.mean([p['спрос'] for p in prognoz])
total   = sum(p['спрос'] for p in prognoz)
max_day = max(prognoz, key=lambda x: x['спрос'])
min_day = min(prognoz, key=lambda x: x['спрос'])

print(f'\n{sep}')
print('  ИТОГИ ПРОГНОЗА НА МЕСЯЦ')
print(sep)
print(f'  Среднесуточный спрос  : {avg_p:.1f} стак./день')
print(f'  Итого за 30 дней      : {int(total)} стаканчиков')
print(f'  Максимум              : {int(max_day["спрос"])} стак. (день {max_day["день"]}, {max_day["температура"]}°C)')
print(f'  Минимум               : {int(min_day["спрос"])} стак. (день {min_day["день"]}, {min_day["температура"]}°C)')

# ── Простая линейная регрессия (сравнительная) ───────────────
x_s    = df['price'].values.astype(float)
y_s    = df['demand'].values.astype(float)
X_s    = np.column_stack([np.ones(len(y_s)), x_s])
b_s    = np.linalg.lstsq(X_s, y_s, rcond=None)[0]
yhat_s = X_s @ b_s
res_s  = y_s - yhat_s
R2_s   = 1 - np.sum(res_s**2)/np.sum((y_s-y_s.mean())**2)
mape_s = np.mean(np.abs(res_s/y_s)) * 100

print(f'\n{SEP}')
print('  СРАВНИТЕЛЬНАЯ МОДЕЛЬ: ПРОСТАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ')
print('  (один фактор: цена)')
print(SEP)
sign_b = '+' if b_s[1] >= 0 else '-'
print(f'  Уравнение: Спрос = {b_s[0]:.3f} {sign_b} {abs(b_s[1]):.5f}*Цена')
print(f'\n{sep}')
print(f'  {"Показатель":<35} {"МЛР":>15} {"Простая":>15}')
print('  ' + '-'*66)
print(f'  {"R2 (коэфф. детерминации)":<35} {R2:>15.4f} {R2_s:>15.4f}')
print(f'  {"MAPE (ошибка аппроксимации)":<35} {mape:>14.2f}% {mape_s:>14.2f}%')
print(f'  {"Число факторов":<35} {k:>15} {"1":>15}')
print('  ' + '-'*66)
print(f'  Вывод: МЛР лучше — R2 выше на {R2-R2_s:.4f}, MAPE ниже на {mape_s-mape:.2f}%')
print(f'\n  Датасет сохранён: {csv_path}')
print(SEP)

# ═══════════════════════════════════════════════════════════════
# FLASK-ПРИЛОЖЕНИЕ — СРАВНЕНИЕ ДВУХ КОФЕЕН
# ═══════════════════════════════════════════════════════════════

import webbrowser
import threading
import logging

logging.getLogger('werkzeug').setLevel(logging.ERROR)

try:
    from flask import Flask, render_template_string, request, jsonify
except ImportError:
    print('Установите Flask: pip install flask')
    exit()

app = Flask(__name__)

FACTOR_LABELS = {
    'price':       'Цена на стаканчик (бел. руб.)',
    'advertising': 'Наличие рекламы',
    'traffic':     'Пешеходный трафик (чел/день)',
    'hours':       'Время работы (часов в день)',
    'rating':      'Рейтинг кофейни (1.0 - 5.0)',
    'competitors': 'Число конкурентов рядом',
}

DEFAULTS_A = {'price':5.5,'advertising':1,'traffic':1100,'hours':13,'rating':4.4,'competitors':2}
DEFAULTS_B = {'price':8.9,'advertising':0,'traffic':480, 'hours':10,'rating':3.7,'competitors':5}
FMIN  = {'price':2.0, 'advertising':0,'traffic':0,   'hours':1, 'rating':1.0,'competitors':0}
FMAX  = {'price':20.0,'advertising':1,'traffic':5000,'hours':24,'rating':5.0,'competitors':15}
FSTEP = {'price':0.1, 'advertising':1,'traffic':50,  'hours':0.5,'rating':0.1,'competitors':1}

factor_ru = {
    'price':'цена','advertising':'реклама','traffic':'трафик',
    'hours':'время работы','rating':'рейтинг','competitors':'число конкурентов'
}

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Сравнение кофеен</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f4f8;color:#2d3748;min-height:100vh}

header{background:linear-gradient(135deg,#2b6cb0,#3182ce,#63b3ed);padding:26px 40px 22px;box-shadow:0 2px 8px rgba(49,130,206,0.25)}
header h1{font-size:1.5rem;font-weight:700;color:#fff}
header p{margin-top:5px;color:#bee3f8;font-size:0.82rem}

.stats-bar{display:flex;flex-wrap:wrap;gap:9px;padding:14px 40px;background:#fff;border-bottom:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
.chip{background:#ebf8ff;border:1px solid #bee3f8;border-radius:20px;padding:4px 13px;font-size:0.76rem;color:#2c5282}
.chip span{color:#2b6cb0;font-weight:700}

.model-bar{display:flex;align-items:center;gap:12px;padding:12px 40px;background:#fff;border-bottom:1px solid #e2e8f0}
.model-bar label{font-size:0.84rem;color:#4a5568;font-weight:600}
.model-bar select{padding:7px 32px 7px 12px;border:1px solid #bee3f8;border-radius:8px;background:#ebf8ff;color:#2c5282;font-size:0.84rem;cursor:pointer;outline:none;font-family:inherit;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath fill='%232b6cb0' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center}
.mbadge{padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:600}
.mb-mlr{background:#c6f6d5;color:#276749;border:1px solid #9ae6b4}
.mb-sim{background:#fefcbf;color:#744210;border:1px solid #f6e05e}

.main{display:grid;grid-template-columns:1fr 1fr;max-width:1300px;margin:24px auto 0;padding:0 32px}
.col{padding:0 20px}
.col:first-child{border-right:1px solid #e2e8f0;padding-left:0}
.col:last-child{padding-right:0}
.col-hdr{display:flex;align-items:center;gap:10px;margin-bottom:20px}
.dot{width:10px;height:10px;border-radius:50%}
.dot-a{background:#c6893a;box-shadow:0 0 6px #c6893a44}
.dot-b{background:#3182ce;box-shadow:0 0 6px #3182ce44}
.col-hdr h2{font-size:0.95rem;font-weight:600;color:#2d3748}

.row{margin-bottom:15px}
.row label{display:flex;justify-content:space-between;font-size:0.78rem;color:#718096;margin-bottom:5px}
.row label .v{font-weight:700;color:#2d3748}

input[type=range]{width:100%;height:4px;-webkit-appearance:none;background:#bee3f8;border-radius:2px;outline:none;cursor:pointer}
input[type=range].a::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;background:#c6893a;box-shadow:0 1px 4px rgba(0,0,0,0.15);cursor:pointer}
input[type=range].b::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;background:#3182ce;box-shadow:0 1px 4px rgba(0,0,0,0.15);cursor:pointer}

.toggle{display:flex;gap:7px;margin-top:4px}
.tbtn{flex:1;padding:7px;border:1px solid #e2e8f0;background:#f7fafc;color:#a0aec0;border-radius:8px;cursor:pointer;font-size:0.78rem;transition:all 0.15s}
.tbtn.on-a{background:#fef3e2;border-color:#c6893a;color:#92600a;font-weight:600}
.tbtn.on-b{background:#ebf8ff;border-color:#3182ce;color:#2c5282;font-weight:600}

.results{max-width:1300px;margin:22px auto 40px;padding:0 32px}
.res-lbl{font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;color:#a0aec0;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #e2e8f0}

.cards{display:grid;grid-template-columns:1fr 1fr 1fr;gap:13px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:22px;text-align:center;transition:transform 0.2s,box-shadow 0.2s;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
.card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.08)}
.card.win-a{border-color:#c6893a;background:#fef9f0}
.card.win-b{border-color:#3182ce;background:#ebf8ff}
.card .lbl{font-size:0.72rem;color:#a0aec0;margin-bottom:7px}
.card .big{font-size:2.4rem;font-weight:700;line-height:1}
.ca{color:#c6893a}.cb{color:#3182ce}.cp{color:#276749}.cn{color:#c53030}
.card .unit{font-size:0.72rem;color:#cbd5e0;margin-top:4px}
.badge{display:inline-block;margin-top:9px;padding:3px 12px;border-radius:20px;font-size:0.7rem;font-weight:600}
.ba{background:#fef3e2;border:1px solid #c6893a;color:#92600a}
.bb{background:#ebf8ff;border:1px solid #3182ce;color:#2c5282}
.bn{background:#f7fafc;border:1px solid #e2e8f0;color:#a0aec0}

.verdict{margin-top:13px;padding:14px 18px;background:#fff;border-radius:10px;border-left:4px solid #3182ce;font-size:0.84rem;color:#4a5568;line-height:1.65;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
.verdict strong{color:#2b6cb0}

.dim{opacity:0.3;pointer-events:none}
footer{text-align:center;padding:14px;color:#cbd5e0;font-size:0.7rem;margin-top:10px}
</style>
</head>
<body>

<header>
  <h1>☕ Сравнение кофеен — прогноз суточного спроса</h1>
  <p>МЛР и простая линейная регрессия (по цене) · 60 кофеен · белорусские рубли</p>
</header>

<div class="stats-bar">
  <div class="chip">R² МЛР = <span>{{ r2 }}</span></div>
  <div class="chip">MAPE МЛР = <span>{{ mape }}%</span></div>
  <div class="chip">R² Простой = <span>{{ r2s }}</span></div>
  <div class="chip">MAPE Простой = <span>{{ mapes }}%</span></div>
  <div class="chip">Харке-Бера p = <span>{{ jb_p }}</span></div>
  <div class="chip">Тест Уайта p = <span>{{ wh_p }}</span></div>
  <div class="chip">DW = <span>{{ dw }}</span></div>
</div>

<div class="model-bar">
  <label>Модель:</label>
  <select id="modelSel" onchange="changeModel()">
    <option value="mlr">МЛР — множественная линейная регрессия (6 факторов)</option>
    <option value="simple">Простая линейная регрессия (фактор: цена)</option>
  </select>
  <span class="mbadge mb-mlr" id="mbadge">R² = {{ r2 }}</span>
</div>

<div class="main">
  <div class="col">
    <div class="col-hdr"><div class="dot dot-a"></div><h2>Кофейня А</h2></div>
    {% for f in factors %}
    <div class="row" id="rowA_{{ f }}">
      {% if f == 'advertising' %}
      <label>{{ labels[f] }} <span class="v" id="va_{{ f }}">Есть</span></label>
      <div class="toggle">
        <button class="tbtn on-a" id="ta1" onclick="setAdv('a',1)">✓ Есть</button>
        <button class="tbtn"      id="ta0" onclick="setAdv('a',0)">✗ Нет</button>
      </div>
      {% else %}
      <label>{{ labels[f] }} <span class="v" id="va_{{ f }}">{{ da[f] }}</span></label>
      <input type="range" class="a" id="ra_{{ f }}"
             min="{{ mn[f] }}" max="{{ mx[f] }}" step="{{ st[f] }}"
             value="{{ da[f] }}" oninput="upd('a','{{ f }}',this.value)">
      {% endif %}
    </div>
    {% endfor %}
  </div>

  <div class="col">
    <div class="col-hdr"><div class="dot dot-b"></div><h2>Кофейня Б</h2></div>
    {% for f in factors %}
    <div class="row" id="rowB_{{ f }}">
      {% if f == 'advertising' %}
      <label>{{ labels[f] }} <span class="v" id="vb_{{ f }}">Нет</span></label>
      <div class="toggle">
        <button class="tbtn"      id="tb1" onclick="setAdv('b',1)">✓ Есть</button>
        <button class="tbtn on-b" id="tb0" onclick="setAdv('b',0)">✗ Нет</button>
      </div>
      {% else %}
      <label>{{ labels[f] }} <span class="v" id="vb_{{ f }}">{{ db[f] }}</span></label>
      <input type="range" class="b" id="rb_{{ f }}"
             min="{{ mn[f] }}" max="{{ mx[f] }}" step="{{ st[f] }}"
             value="{{ db[f] }}" oninput="upd('b','{{ f }}',this.value)">
      {% endif %}
    </div>
    {% endfor %}
  </div>
</div>

<div class="results">
  <div class="res-lbl">Результаты прогноза</div>
  <div class="cards">
    <div class="card" id="cA">
      <div class="lbl">Кофейня А</div>
      <div class="big ca" id="rA">—</div>
      <div class="unit">стаканчиков / день</div>
    </div>
    <div class="card" id="cB">
      <div class="lbl">Кофейня Б</div>
      <div class="big cb" id="rB">—</div>
      <div class="unit">стаканчиков / день</div>
    </div>
    <div class="card" id="cD">
      <div class="lbl">Разница (А − Б)</div>
      <div class="big" id="rD">—</div>
      <div class="unit">стаканчиков / день</div>
      <div class="badge bn" id="wbadge">—</div>
    </div>
  </div>
  <div class="verdict" id="verdict">Настройте параметры кофеен, чтобы увидеть прогноз.</div>
</div>

<footer>Курсовая работа · Моделирование поведения потребителей · МЛР · Python + Flask</footer>

<script>
const factors = {{ factors|tojson }};
const adv = {a:1, b:0};
let curModel = 'mlr';

function changeModel(){
  curModel = document.getElementById('modelSel').value;
  const badge = document.getElementById('mbadge');
  if(curModel === 'mlr'){
    badge.textContent = 'R\u00B2 = {{ r2 }}';
    badge.className = 'mbadge mb-mlr';
    factors.forEach(f => {
      document.getElementById('rowA_'+f).classList.remove('dim');
      document.getElementById('rowB_'+f).classList.remove('dim');
    });
  } else {
    badge.textContent = 'R\u00B2 = {{ r2s }}';
    badge.className = 'mbadge mb-sim';
    factors.forEach(f => {
      if(f !== 'price'){
        document.getElementById('rowA_'+f).classList.add('dim');
        document.getElementById('rowB_'+f).classList.add('dim');
      }
    });
  }
  predict();
}

function setAdv(c,v){
  adv[c]=v;
  document.getElementById('v'+c+'_advertising').textContent=v?'Есть':'Нет';
  const cls=c==='a'?'on-a':'on-b';
  document.getElementById('t'+c+'1').className='tbtn'+(v===1?' '+cls:'');
  document.getElementById('t'+c+'0').className='tbtn'+(v===0?' '+cls:'');
  predict();
}
function upd(c,f,v){
  document.getElementById('v'+c+'_'+f).textContent=v;
  predict();
}
function getVals(c){
  const o={};
  for(const f of factors){
    if(f==='advertising') o[f]=adv[c];
    else o[f]=parseFloat(document.getElementById('r'+c+'_'+f).value);
  }
  return o;
}
async function predict(){
  const res = await fetch('/predict',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({a:getVals('a'),b:getVals('b'),model:curModel})
  });
  const d = await res.json();
  document.getElementById('rA').textContent = d.da;
  document.getElementById('rB').textContent = d.db;
  const diff = d.da - d.db;
  const dEl = document.getElementById('rD');
  dEl.textContent = (diff>=0?'+':'')+diff;
  dEl.className = 'big '+(diff>0?'cp':(diff<0?'cn':'ca'));
  const wb = document.getElementById('wbadge');
  const cA = document.getElementById('cA');
  const cB = document.getElementById('cB');
  cA.className='card'; cB.className='card';
  if(diff>0){wb.textContent='🏆 Побеждает А';wb.className='badge ba';cA.className='card win-a';}
  else if(diff<0){wb.textContent='🏆 Побеждает Б';wb.className='badge bb';cB.className='card win-b';}
  else{wb.textContent='🤝 Ничья';wb.className='badge bn';}
  document.getElementById('verdict').innerHTML = d.verdict;
}
predict();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML,
        factors=factors_keep,
        labels=FACTOR_LABELS,
        da=DEFAULTS_A, db=DEFAULTS_B,
        mn=FMIN, mx=FMAX, st=FSTEP,
        r2=f'{R2:.4f}', mape=f'{mape:.2f}',
        r2s=f'{R2_s:.4f}', mapes=f'{mape_s:.2f}',
        jb_p=f'{jb_p:.4f}', wh_p=f'{wh_p:.4f}', dw=f'{dw:.4f}'
    )

@app.route('/predict', methods=['POST'])
def predict_route():
    data  = request.get_json()
    model = data.get('model','mlr')

    if model == 'mlr':
        def calc(vals):
            x = np.array([vals[f] for f in factors_keep])
            return max(0, round(float(np.dot(beta, np.concatenate([[1],x]))),1))
        da   = calc(data['a'])
        db   = calc(data['b'])
        diff = round(da-db, 1)
        contrib = {f: abs(data['a'].get(f,0)-data['b'].get(f,0))*abs(beta[i+1])
                   for i,f in enumerate(factors_keep)}
        top = max(contrib, key=contrib.get)
        if abs(diff) < 1:
            verdict = 'Кофейни показывают <strong>практически одинаковый прогноз</strong> по модели МЛР.'
        elif diff > 0:
            verdict = (f'<strong>Кофейня А</strong> опережает Б на <strong>{abs(diff)} стак./день</strong> (МЛР). '
                       f'Главный фактор: <strong>«{factor_ru.get(top,top)}»</strong>. '
                       f'За месяц разница ~<strong>{round(abs(diff)*30)}</strong> стаканчиков.')
        else:
            verdict = (f'<strong>Кофейня Б</strong> опережает А на <strong>{abs(diff)} стак./день</strong> (МЛР). '
                       f'Главный фактор: <strong>«{factor_ru.get(top,top)}»</strong>. '
                       f'За месяц разница ~<strong>{round(abs(diff)*30)}</strong> стаканчиков.')
    else:
        def calc_s(vals):
            return max(0, round(float(b_s[0] + b_s[1]*float(vals.get('price',0))),1))
        da   = calc_s(data['a'])
        db   = calc_s(data['b'])
        diff = round(da-db, 1)
        if abs(diff) < 1:
            verdict = 'Кофейни показывают <strong>практически одинаковый прогноз</strong> по простой регрессии.'
        elif diff > 0:
            verdict = (f'<strong>Кофейня А</strong> опережает Б на <strong>{abs(diff)} стак./день</strong> (простая регрессия, фактор: цена). '
                       f'R² = {R2_s:.4f} — точность ниже МЛР.')
        else:
            verdict = (f'<strong>Кофейня Б</strong> опережает А на <strong>{abs(diff)} стак./день</strong> (простая регрессия, фактор: цена). '
                       f'R² = {R2_s:.4f} — точность ниже МЛР.')

    return jsonify(da=da, db=db, verdict=verdict)

def open_browser():
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print(f"\n{'='*64}")
    print('  Запускаю приложение — открываю браузер...')
    print('  Адрес: http://127.0.0.1:5000')
    print(f"{'='*64}\n")
    threading.Timer(1.2, open_browser).start()
    app.run(debug=False, use_reloader=False)
