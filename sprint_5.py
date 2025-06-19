
from readline import redisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy import stats as st
from math import factorial
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import ttest_ind



megaline_calls = pd.read_csv("/content/megaline_calls.csv")
megaline_internet = pd.read_csv("/content/megaline_internet.csv")
megaline_messages = pd.read_csv("/content/megaline_messages.csv")
megaline_plans = pd.read_csv("/content/megaline_plans.csv")
megaline_users = pd.read_csv("/content/megaline_users.csv")


redisplay(megaline_users.sample(5))

redisplay(megaline_internet.sample(5))

redisplay(megaline_calls.sample(5))

redisplay(megaline_messages.sample(5))

redisplay(megaline_plans)
megaline_plans.info()


megaline_plans.rename(columns={'plan_name' : 'plan'}, inplace = True)

megaline_plans.rename(columns={'mb_per_month_included': 'gb_per_month_included'}, inplace =True)
megaline_plans['gb_per_month_included'] = megaline_plans['gb_per_month_included'] / 1024
print(megaline_plans)


megaline_users.info()
print("""
cantidad de usuarios registrados en surf:""", megaline_users['user_id'][megaline_users['plan'] == 'surf'].nunique(),
"""
cantidad de usuarios registrados en ultimate:""", megaline_users['user_id'][megaline_users['plan'] == 'ultimate'].nunique())

print("cantidad de datos duplicados: ", megaline_users.duplicated().sum())
print("""
cantidad de datos ausentes:
""",megaline_users.isna().sum())
megaline_users['age'].describe()

megaline_users.sample(5)

"""La edad minima en los usuarios es de 18, la máxima de 75 y la mediana es de 46 que es casi igual a la media (45)

Los 'users' activos en el plan aparecen en la tabla como NaN, que es necesario reemplazar para evitar confusiones.

### Corregir los datos
"""

megaline_users['churn_date'] = megaline_users['churn_date'].fillna('active')
megaline_users['churn_date'].isna().sum()
print(megaline_users.sample(5))



megaline_calls.info()

print("cantidad datos duplicados: ", megaline_calls.duplicated().sum())
print("""
cantidad datos ausentes:
""", megaline_calls.isna().sum())
megaline_calls['duration'].describe()

megaline_calls.sample(5)

"""Registro de más de 137000 llamadas registradas de 481 usuarios.

No se observan datos nulos o ausentes.

Las llamadas que se observan registradas con 00 sugiere que son de segundos y no llegan al minuto

"""

megaline_calls['user_id'].nunique()
megaline_calls['call_date'] = pd.to_datetime(megaline_calls['call_date'])
megaline_calls['month'] = megaline_calls['call_date'].dt.month
megaline_calls['day'] = megaline_calls['call_date'].dt.day
print(megaline_calls.sample(5))



megaline_messages.info()
print('cantidad de usuarios que enviaron al menos un sms: ', megaline_messages['user_id'].nunique())
print("cantidad datos duplicados: ", megaline_messages.duplicated().sum())
print("""cantidad datos ausentes:
""", megaline_messages.isna().sum())

megaline_messages.sample(5)

"""Más de 76000 registros de mensajes de 402 usuarios.

No se observan datos nulos o ausentes, tampoco errores o datos confusos.


"""

megaline_messages['message_date'] = pd.to_datetime(megaline_messages['message_date'])
megaline_messages['month'] = megaline_messages['message_date'].dt.month
megaline_messages['day'] = megaline_messages['message_date'].dt.day
print(megaline_messages.sample(5))


megaline_internet.info()
print('cantidad de usuarios que consumieron al menos un MB: ', megaline_internet['user_id'].nunique())
print("cantidad datos duplicados: ", megaline_internet.duplicated().sum())
print("""cantidad datos ausentes:
""", megaline_internet.isna().sum())
megaline_internet['mb_used'].describe()

megaline_internet.sample(5)

"""Se obtuvieron 104000+ registros de 489 usuarios.

No se observan datos duplicados o ausentes.

se hace la conversión: 1000MB = 1GB

La media de uso de MB es de 366 (0.366GB), con una mediana de 343 (0.343) y una máxima de 1693 (1.693GB).

"""

megaline_internet.rename(columns={'mb_used': 'gb_used'}, inplace=True)
megaline_internet['gb_used'] = megaline_internet['gb_used'] / 1024



megaline_internet['session_date'] = pd.to_datetime(megaline_internet['session_date'])
megaline_internet['month'] = megaline_internet['session_date'].dt.month
megaline_internet['day'] = megaline_internet['session_date'].dt.day
print(megaline_internet.sample(5))



print(megaline_plans)

"""## Agrego datos por usuario

"""

user_calls_month = megaline_calls.groupby(['user_id', 'month'])['id'].count().reset_index()

user_calls_month.rename(columns={'id': 'calls_amount'}, inplace=True)
print(user_calls_month.sample(10))

user_calls_duration_month = megaline_calls.groupby(['user_id', 'month'])['duration'].sum()
print(user_calls_duration_month)

user_sms_month = megaline_messages.groupby(['user_id', 'month'])['id'].count().reset_index()

user_sms_month.rename(columns={'id': 'amount_sms'}, inplace=True)
print(user_sms_month)

user_gb = megaline_internet.groupby(['user_id', 'month'])['gb_used'].sum()
print(user_gb)

user_data = pd.merge(user_calls_month, user_calls_duration_month, on=['user_id', 'month'])
user_data = pd.merge(user_data, user_sms_month, on=['user_id', 'month'])
user_data = pd.merge(user_data, user_gb, on=['user_id', 'month'])
user_data['gb_used'] = user_data['gb_used'].round(2)
print(user_data)

sum_df = pd.merge(user_data, megaline_users[['user_id', 'plan']], how='left')
sum_df = sum_df.reset_index(drop=True)
print(sum_df)

user_data = pd.merge(user_data, megaline_users[['user_id', 'plan']], on='user_id')
user_data = pd.merge(user_data, megaline_plans, on='plan')
user_data = user_data.iloc[:, [0, 1, 6, 10, 9, 2, 3, 13, 7, 4, 12, 8, 5, 11]]
#print(user_data)

user_data['minutes_not_included'] =  user_data['duration'] - user_data['minutes_included']
user_data['sms_not_included'] = user_data['amount_sms'] - user_data['messages_included']
user_data['gb_not_included'] =  user_data['gb_used'] - user_data['gb_per_month_included']
user_data['minutes_to_pay'] = user_data['minutes_not_included'] * user_data['usd_per_minute']
user_data['sms_to_pay'] = user_data['sms_not_included'] * user_data['usd_per_message']
user_data['gb_to_pay'] = user_data['gb_not_included'] * user_data['usd_per_gb']
user_data = user_data.iloc[:, [ 0, 1, 2, 3, 4, 5, 6, 7, 14, 17, 8, 9, 10, 15, -2, 11, 12, 13, -4, -1]]
user_data['minutes_to_pay'] = user_data['minutes_to_pay'].round(2)

#print(user_data)

def negative_number(x):
    if x < 0:
        return 0
    else:
        return x

user_data['minutes_to_pay'] = user_data['minutes_to_pay'].apply(negative_number)
user_data['sms_to_pay'] = user_data['sms_to_pay'].apply(negative_number)
user_data['gb_to_pay'] = user_data['gb_to_pay'].apply(negative_number)
user_data['monthly_income'] = user_data['minutes_to_pay'] + user_data['sms_to_pay'] + user_data['gb_to_pay']
print(user_data)

incomes = user_data[['user_id','month', 'duration', 'amount_sms', 'gb_used','plan','monthly_income']]
print(incomes)

surf_users = user_data[user_data['plan'] == 'surf']
ultimate_users = user_data[user_data['plan'] == 'ultimate']

#print(surf_users)
#print(ultimate_users)


# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla.
calls_media = megaline_calls.merge(megaline_users[['user_id', 'plan']], on='user_id')
calls_media_month = calls_media.pivot_table(index='month', columns='plan', values='duration', aggfunc='mean')

calls_media_month.index=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ags', 'Sept', 'Oct', 'Nov', 'Dic']
print(calls_media_month.round(2))

calls_media_month.plot(title='Promedio de llamadas por cada plan y por cada mes', kind='bar', ylabel='Duración promedio en minutos',  xlabel="meses" ,rot=65, figsize=[10, 4])
plt.show()

# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.
surf_duration = user_data[user_data['plan'] == 'surf'].groupby(['user_id', 'month'])['duration'].sum()
ultimate_duration = user_data[user_data['plan'] == 'ultimate'].groupby(['user_id', 'month'])['duration'].sum()

surf_duration.hist(bins=30, color='blue', label='Surf', figsize=[20, 5])
ultimate_duration.hist(bins=30, alpha=0.9, color='orange', label='Ultimate')
plt.legend(['Surf', 'Ultimate'])
plt.show()

# Calcula la media y la varianza de la duración mensual de llamadas.

calls_var_month = calls_media_month.var()
calls_media_month_mean = calls_media_month.mean()

#monthly_call_mean = user_data['duration'].mean() primer intento
#monthly_call_var = user_data['duration'].var() primer intento

#monthly_call_mean = megaline_calls.groupby(['month', 'plan'])['duration'].mean()
#monthly_call_var = megaline_calls.groupby(['month', 'plan'])['duration'].var()
#print(monthly_call_var, monthly_call_mean)

print('la media por mes es ',calls_media_month_mean.round(2))
print('la varianza por mes es ', calls_var_month.round(2))

# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas
print('Distribución de la duracion mensual de llamadas por mes por usuario de cada plan')
sns.boxplot(data=user_data, x='plan', y='duration')
plt.show()

#print('Distribucion de la duracion promedio de llamadas por mes')
#sns.boxplot(data=calls_media_month)
plt.show()

#calls_media_month

#sms_media = megaline_messages.merge(megaline_users[['user_id', 'plan']], on='user_id')
sms_media_month = user_data.pivot_table(index='month', columns='plan', values='amount_sms', aggfunc='sum')

sms_media_month.index=['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
sms_media_month.plot(title='Promedio de mensajes por plan y por mes', kind='bar', ylabel='mensajes',  xlabel="meses" ,rot=65, figsize=[10, 4])
plt.show()

sms_media_month = user_data.pivot_table(index='month', columns='plan', values='amount_sms', aggfunc='mean')

sms_media_month.index=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Agos', 'Sept', 'Oct', 'Nov', 'Dic']
sms_media_month.plot(title='Promedio de mensajes por plan y por mes', kind='bar', ylabel='mensajes',  xlabel="meses" ,rot=65, figsize=[10, 4])
plt.show()

# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
surf_sms = user_data[user_data['plan'] == 'surf'].groupby(['user_id', 'month'])['amount_sms'].sum()
ultimate_sms = user_data[user_data['plan'] == 'ultimate'].groupby(['user_id', 'month'])['amount_sms'].sum()

surf_sms.plot(kind='hist', bins=25, alpha=0.8, color='blue', label='Surf', figsize=[12,6])

ultimate_sms.plot(kind='hist',bins=25, alpha=0.8, color='orange', label='Ultimate')
plt.legend(['Surf', 'Ultimate'])
plt.show()

sns.boxplot(data=sms_media_month)
plt.show()


internet_total_month = user_data.pivot_table(index='month', columns='plan', values='gb_used', aggfunc='sum')

internet_total_month.index=['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
internet_total_month.plot(title='Promedio de GBs por cada plan y por cada mes', kind='bar', ylabel='Duración promedio en minutos',  xlabel="meses" ,rot=65, figsize=[10, 4])
plt.show()

internet_media_month = user_data.pivot_table(index='month', columns='plan', values='gb_used', aggfunc='mean')

internet_media_month.index=['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
internet_media_month.plot(title='Promedio de GBs por cada plan y por cada mes', kind='bar', ylabel='Gigabites',  xlabel="meses" ,rot=65, figsize=[10, 4])
plt.show()

print(internet_media_month.round(2))
redisplay(internet_media_month.describe())

surf_internet = user_data[user_data['plan'] == 'surf'].groupby(['user_id', 'month'])['gb_used'].sum()
ultimate_internet = user_data[user_data['plan'] == 'ultimate'].groupby(['user_id', 'month'])['gb_used'].sum()


surf_internet.plot(kind='hist', bins=30, alpha=0.8, color='blue', label='Surf', figsize=[12,6])
ultimate_internet.plot(kind='hist',bins=30, alpha=0.8, color='orange', label='Ultimate')
plt.legend(['Surf', 'Ultimate'])
plt.show()

sns.boxplot(data=internet_media_month)
plt.show()


ingreso_media_month = user_data.pivot_table(index='month', columns='plan', values='monthly_income', aggfunc='sum')


ingreso_media_month.index=['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
print(ingreso_media_month)
ingreso_media_month.plot(title='Promedio de llamadas por cada plan y por cada mes', kind='bar', ylabel='Duración promedio en minutos',  xlabel="meses" ,rot=65, figsize=[10, 4])
plt.show()

surf_ingreso = user_data[user_data['plan'] == 'surf'].groupby(['user_id', 'month'])['monthly_income'].sum()
ultimate_ingreso = user_data[user_data['plan'] == 'ultimate'].groupby(['user_id', 'month'])['monthly_income'].sum()

surf_ingreso.plot(kind='hist', bins=30, alpha=0.8, color='blue', label='Surf', figsize=[12,6])
ultimate_ingreso.plot(kind='hist',bins=30, alpha=0.8, color='orange', label='Ultimate')
plt.legend(['Surf', 'Ultimate'])
plt.show()

sns.boxplot(data=internet_media_month)
plt.show()



# Prueba las hipótesis
# Hipotesis nula: Los ingresos promedio procedentes de los usuarios son iguales entro los planes Ultimate y Surf
# Hipotesis alternativa: Los ingresos promedio procedentes de los usuarios son diferentes entro los planes Ultimate y Surf
alfa = 0.05
surf_ingreso = user_data[user_data['plan'] == 'surf'].groupby(['user_id'])['monthly_income'].mean()
ultimate_ingreso = user_data[user_data['plan'] == 'ultimate'].groupby(['user_id'])['monthly_income'].mean()
#resultado = st.ttest_ind(surf_ingreso, ultimate_ingreso, equal_var=False)

t_stat, p_val_2tail = st.ttest_ind(surf_ingreso, ultimate_ingreso, equal_var=False)


print(f't_statistic: {t_stat}')
print(f'p_value: {p_val_2tail}')

#if resultado.pvalue < alfa:
if p_val_2tail < alfa:
    print('Rechazamos la hipótesis nula')
else:
    print('No rechazamos la hipótesis nula')


# Prueba las hipótesis
#H0 o nula = el ingreso promedio de los usuarios del área NY-NJ es igual al de los usuarios de otras regiones.
#H1 o alternativa = el ingreso promedio de los usuarios del área NY-NJ es diferente al de los usuarios de otras regiones.
alfa = 0.05
user_data_city = user_data.merge(megaline_users[['user_id', 'city']], on='user_id')
NYNJ_ingreso = user_data_city.query('city.str.contains("NY-NJ")').groupby(['user_id'])['monthly_income'].mean()
otros_ingreso = user_data_city.query('not city.str.contains("NY-NJ")').groupby(['user_id'])['monthly_income'].mean()
t_stat, p_val_2tail = st.ttest_ind(NYNJ_ingreso, otros_ingreso, equal_var=False)
print(f'p-valor para 2 colas: {p_val_2tail}')

#p_val_1tail = p_val_2tail / 2
#print(f'p-valor para 1 cola: {p_val_1tail}')

if p_val_2tail < alfa:
    print('Rechazamos la hipótesis nula')
else:
    print('No rechazamos la hipótesis nula, no hay suficientes evidencias para concluir que el ingreso promedio de los usuarios del area de NY-NJ es diferente al de los usuarios de otras regiones')

"""## Conclusión general

Se recibe una muestra de la empresa 'Megaline' del año 2018 donde se registraron 500 usuarios, con el fin de poder 'saber cuál de las tarifas genera más ingresos para poder ajustar el presupuesto de publicidad'.
Una vez observados los datos, primero se corrigieron ciertos errores: nombres en columnas para poder combinarlas más adelante, también se segmentaron los datos en meses, para poder entender mejor cómo fue el comportamiento de los clientes a lo largo del año.
Primera observación general:

Se observa mayor cantidad de usuarios del plan Surf.
Se observa un crecimiento en ambos planes en cuanto a la cantidad de actividad por mes, esto puede referirse a nuevos clientes.
Primeras conclusiones en cuanto a llamadas:

La cantidad promedio de llamadas para ambos usuarios es entre 6.4 y 6.8 minutos. Es decir, no hay diferencias significativas.
La distribución de los datos en cuanto al número utilizado de minutos mensuales por usuario es similar.
Primeras conclusiones en cuanto a mensajes:

Al hacer un promedio de qué usuarios envían mayor cantidad de mensajes por mes estos corresponden a los clientes de Ultimate.
El comportamiento de los usuarios en cuanto al número de mensajes que tiende a enviar por mes es similar.
Primeras conclusiones en cuanto a al uso de GBs:

La cantidad de GBs utilizadas por usuario por mes tiene una media similar en cada plan, pero los usuarios de Ultimate utilizan mayor volumen de internet mismo. Se observa gran consumo de los mismos en los meses de febrero y enero.
El comportamiento en al distribucion de los datos es similar.
La media de volumen de internet de Surf es de 14GB por mes, mientras que Ultimate es de 16GB.
Conclusiones en cuanto a ingresos: Con un 95% de certeza, los usuarios de Surf generan mayor cantidad de ingresos en comparación a los de Ultimate. Esto se corrobora con gráficos y una hipótesis nula y otra alternativa.
"""