#!/usr/bin/env python
# coding: utf-8

# In[269]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import datetime


# In[270]:


df = pd.read_csv('taxigov-corridas-7-dias-vF.csv')


# In[271]:


df.head()


# In[272]:


df.shape


# In[273]:


df.dtypes


# In[274]:


df.describe()


# In[275]:


# Como a coluna conteste_info está vazia, ela será deletada


# In[276]:


del df["conteste_info"]


# In[277]:


df.shape


# In[278]:


# Substituição do valor vazio da coluna qru_corrida 


# In[279]:


df['qru_corrida'].fillna(784630.0, inplace=True)
df.describe() # Para conferir a substituição


# In[280]:


# Preenchendo os valores vazios das colunas km_total e valor_corrida


# In[281]:


df['km_total'].replace([np.inf, -np.inf], np.nan, inplace=True) 
df['km_total'].fillna(0.0, inplace=True)

df['valor_corrida'].replace([np.inf, -np.inf], np.nan, inplace=True) 
df['valor_corrida'].fillna(0.0, inplace=True)


# In[282]:


df['km_total'].describe()


# In[283]:


df['valor_corrida'].describe()


# In[284]:


# Preenchendo os valores vazios das colunas que faltam


# In[285]:


df['destino_solicitado_latitude'].replace([np.inf, -np.inf], np.nan, inplace=True) 
df['destino_solicitado_latitude'].fillna(0.0, inplace=True)

df['destino_solicitado_longitude'].replace([np.inf, -np.inf], np.nan, inplace=True) 
df['destino_solicitado_longitude'].fillna(0.0, inplace=True)

df['destino_efetivo_latitude'].replace([np.inf, -np.inf], np.nan, inplace=True) 
df['destino_efetivo_latitude'].fillna(0.0, inplace=True)

df['destino_efetivo_longitude'].replace([np.inf, -np.inf], np.nan, inplace=True) 
df['destino_efetivo_longitude'].fillna(0.0, inplace=True)


# In[286]:


df.describe()


# In[287]:


# Padronizando os dados da coluna motivo_corrida


# In[288]:


df.motivo_corrida.unique()


# In[289]:


df.loc[df['motivo_corrida'] == '01 - Reunião Externa (Ida/Volta)', 'motivo_corrida'] = '1 - REUNIAO EXTERNA'
df.loc[df['motivo_corrida'] == '01 Reunião Externa (Ida/Volta)', 'motivo_corrida'] = '1 - REUNIAO EXTERNA'
df.loc[df['motivo_corrida'] == '02 - Entrega de Documentos', 'motivo_corrida'] = '2 - ENTREGA DE DOCUMENTOS'
df.loc[df['motivo_corrida'] == '02 Entrega de Documentos', 'motivo_corrida'] = '2 - ENTREGA DE DOCUMENTOS'
df.loc[df['motivo_corrida'] == '04 Outros', 'motivo_corrida'] = '4 - OUTROS'
df.loc[df['motivo_corrida'] == '05 - Perícia Médica', 'motivo_corrida'] = '5 - PERICIA MEDICA'
df.loc[df['motivo_corrida'] == '06 - Inspeção/Fiscalização', 'motivo_corrida'] = '6 - INSPECAO/FISCALIZACAO'
df.loc[df['motivo_corrida'] == '06 Fiscalização', 'motivo_corrida'] = '6 - INSPECAO/FISCALIZACAO'
df.loc[df['motivo_corrida'] == '07 - Evento', 'motivo_corrida'] = '7 - EVENTO'
df.loc[df['motivo_corrida'] == '08 - Atendimento Técnico', 'motivo_corrida'] = '8 - ATENDIMENTO TECNICO'
df.loc[df['motivo_corrida'] == '03 - Capacitação/Treinamento', 'motivo_corrida'] = '3 - CAPACITACAO/TREINAMENTO'
df.loc[df['motivo_corrida'] == '10 - Outros', 'motivo_corrida'] = '10 - OUTROS'


# In[290]:


df.motivo_corrida.unique()


# In[291]:


# Analisando as bases de origem


# In[292]:


df.base_origem.unique()


# In[293]:


# Analisandos os motivos das corridas para cada base de origem


# In[294]:


motivos_df_series = df[df.base_origem == 'TAXIGOV_DF'].motivo_corrida.value_counts()
motivos_df_series


# In[295]:


plt.bar(motivos_df_series.keys(), motivos_df_series.array)
plt.xticks(rotation=45)


# In[296]:


motivos_rj_series = df[df.base_origem == 'TAXIGOV_RJ_10'].motivo_corrida.value_counts()
motivos_rj_series


# In[297]:


plt.bar(motivos_rj_series.keys(), motivos_rj_series.array)
plt.xticks(rotation=90)


# In[298]:


motivos_sp_series = df[df.base_origem == 'TAXIGOV_SP_10'].motivo_corrida.value_counts()
motivos_sp_series


# In[299]:


plt.bar(motivos_sp_series.keys(), motivos_sp_series.array)
plt.xticks(rotation=45)


# In[300]:


# Importando a planilha "preco-km-rodado.xlsx"


# In[301]:


df_preco_km = pd.read_excel('preco-km-rodado.xlsx')


# In[302]:


df_preco_km.head()


# In[303]:


# Padronizando a coluna origem_cidade de df para não ter nome diferente da mesma cidade


# In[304]:


df.origem_cidade.unique()


# In[305]:


df.loc[df['origem_cidade'] == 'BRASÍLIA', 'origem_cidade'] = 'BRASILIA'
df.loc[df['origem_cidade'] == 'RJ', 'origem_cidade'] = 'RIO DE JANEIRO'
df.loc[df['origem_cidade'] == 'Rio de Janeiro', 'origem_cidade'] = 'RIO DE JANEIRO'
df.loc[df['origem_cidade'] == 'Duque de Caxias', 'origem_cidade'] = 'DUQUE DE CAXIAS'
df.loc[df['origem_cidade'] == 'Niterói', 'origem_cidade'] = 'NITEROI'
df.loc[df['origem_cidade'] == 'São Caetano do Sul', 'origem_cidade'] = 'SAO CAETANO DO SUL'
df.loc[df['origem_cidade'] == 'São Paulo', 'origem_cidade'] = 'SAO PAULO'


# In[306]:


df.origem_cidade.unique()


# In[307]:


# Unindo os dois dataframes


# In[308]:


df_merged = pd.merge(df, df_preco_km, how='left', left_on='origem_cidade', right_on='Cidade')


# In[309]:


df_merged.head()


# In[310]:


df_merged.describe()


# In[311]:


# Preenhcer os campos vazios das colunas R$/km (bandeira 1), R$/km (bandeira 2), Bandeirada e Tarifa horária com a média


# In[312]:


df_merged['R$/km (bandeira 1)'].fillna(2.867977, inplace=True)
df_merged['R$/km (bandeira 2)'].fillna(3.641700, inplace=True)
df_merged['Bandeirada'].fillna(5.352066, inplace=True)
df_merged['Tarifa horária'].fillna(32.600792, inplace=True)


# In[313]:


# Cálculo do R$/km descontando a bandeirada


# In[314]:


# se km_total = 0 -> valor_por_km_desc_band = 110 (valor acima dos demais que mostrará que é um valor discrepante)
# não posso preencher com a média, por exemplo, pois perderá a informação que procuro


# In[315]:


for i in range(2837):
    if (df_merged.loc[i, 'km_total'] == 0):
        df_merged.loc[i, 'valor_por_km_desc_band'] = 110
    else:
        df_merged.loc[i, 'valor_por_km_desc_band'] = (df_merged.loc[i, 'valor_corrida'] - df_merged.loc[i, 'Bandeirada']) / df_merged.loc[i, 'km_total']


# In[316]:


df_merged.head()


# In[317]:


# Cálculo da diferença do R$/km pago no TáxiGov e o R$/km cobrado pelos táxis em cada cidade na bandeira 1


# In[318]:


df_merged['dif_valorKM_band1'] = df_merged['valor_por_km_desc_band'] - df_merged['R$/km (bandeira 1)']


# In[319]:


# Gráfico do R$/km calculado descontada a bandeirada e o R$/km cobrado pelos táxis


# In[320]:


plt.scatter(df_merged['valor_por_km_desc_band'], df_merged['R$/km (bandeira 1)'])
plt.ylabel('R$/km (bandeira 1)')
plt.xlabel('valor_por_km_desc_band')
plt.show()


# In[321]:


# Eliminando o ponto mais discrepante para enxergar melhor os demais


# In[322]:


filter_vKM = df_merged[(df_merged.valor_por_km_desc_band < 20)]
plt.scatter(filter_vKM['valor_por_km_desc_band'], filter_vKM['R$/km (bandeira 1)'])
plt.ylabel('R$/km (bandeira 1)')
plt.xlabel('valor_por_km_desc_band')
plt.show()


# In[323]:


# Cálculo da distância usando as coordenadas de origem e de destino


# In[324]:


# vectorized haversine function (obtido em: 
# https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows)
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 +         np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


# In[325]:


df_merged['dist_coords'] = haversine(df_merged.origem_latitude, df_merged.origem_longitude, 
                                     df_merged.destino_efetivo_latitude, df_merged.destino_efetivo_longitude)


# In[326]:


df_merged.head()


# In[327]:


df_merged.describe()


# In[328]:


# Cálculo da diferença entre a quilometragem total registrada e a distância obtida pelas coordenadas


# In[329]:


df_merged['dif_distancias'] = df_merged['km_total'] - df_merged['dist_coords']


# In[330]:


# Gráfico dist_coords e km_total


# In[331]:


plt.scatter(df_merged['dist_coords'], df_merged['km_total'])
plt.ylabel('km_total')
plt.xlabel('dist_coords')
plt.show()


# In[332]:


# Eliminando os pontos mais discrepantes para enxergar melhor os demais


# In[333]:


filter_dist = df_merged[(df_merged.dist_coords < 1000)]
plt.scatter(filter_dist['dist_coords'], filter_dist['km_total'])
plt.ylabel('km_total')
plt.xlabel('dist_coords')
plt.show()


# In[334]:


# Gráfico das diferenças de distância e de R$/km


# In[335]:


plt.scatter(df_merged['dif_distancias'], df_merged['dif_valorKM_band1'])
plt.ylabel('dif_valorKM_band1')
plt.xlabel('dif_distancias')
plt.show()


# In[336]:


# Eliminando os pontos mais discrepantes para enxergar melhor os demais


# In[337]:


filter_difs = df_merged[(df_merged.dif_distancias > -100) & (df_merged.dif_distancias < 200) 
                        & (df_merged.dif_valorKM_band1 < 20)]
plt.scatter(filter_difs['dif_distancias'], filter_difs['dif_valorKM_band1'])
plt.ylabel('dif_valorKM_band1')
plt.xlabel('dif_distancias')
plt.show()


# In[338]:


# Criar coluna da % da dif_distancias em relação ao km_total
# se km_total = 0 -> perc_dif_dist = 1000 (valor acima dos demais que mostrará que é um valor discrepante)


# In[339]:


for i in range(2837):
    if (df_merged.loc[i, 'km_total'] == 0):
        df_merged.loc[i, 'perc_dif_dist'] = 1000
    else:
        df_merged.loc[i, 'perc_dif_dist'] = (df_merged.loc[i, 'dif_distancias'] / df_merged.loc[i, 'km_total'])*100


# In[340]:


df_merged.head()


# In[341]:


df_merged.describe()


# In[342]:


# Novo gráfico com dif_valorKM_band1 e perc_dif_dist


# In[421]:


plt.scatter(df_merged['perc_dif_dist'], df_merged['dif_valorKM_band1'])
plt.ylabel('dif_valorKM_band1')
plt.xlabel('perc_dif_dist')
plt.xticks(rotation=90)
plt.show()


# In[344]:


filter_difs_perc = df_merged[(df_merged.perc_dif_dist > -100) & (df_merged.dif_valorKM_band1 < 20)]
plt.scatter(filter_difs_perc['perc_dif_dist'], filter_difs_perc['dif_valorKM_band1'])
plt.ylabel('dif_valorKM_band1')
plt.xlabel('perc_dif_dist')
plt.show()


# In[345]:


# Cálculo da duração da corrida


# In[346]:


df_merged['tempo_corrida'] = pd.to_datetime(df_merged['data_final']) - pd.to_datetime(df_merged['data_inicio'])
t1 = pd.to_datetime(df_merged['data_final'])
t2 = pd.to_datetime(df_merged['data_inicio'])
df_merged['minutos_corrida'] = (df_merged['tempo_corrida']).astype('timedelta64[m]')


# In[347]:


df_merged.describe()


# In[348]:


# Os valores vazios de tempo_corrida e minutos_corrida são por falta de alguma informação, então eles serão preenchidos 
# por zero para mostrar que a informação está faltando


# In[349]:


df_merged['tempo_corrida'].fillna(0.0, inplace=True)
df_merged['minutos_corrida'].fillna(0.0, inplace=True)


# In[350]:


df_merged.describe()


# In[351]:


# Avaliar se as corridas de maior km, maior duração e de maior valor têm R$/km menor


# In[352]:


plt.scatter(df_merged['km_total'], df_merged['valor_por_km_desc_band'])
plt.ylabel('valor por km')
plt.xlabel('km total')
plt.show()


# In[353]:


plt.scatter(df_merged['valor_corrida'], df_merged['valor_por_km_desc_band'])
plt.ylabel('valor por km')
plt.xlabel('valor corrida')
plt.show()


# In[354]:


plt.scatter(df_merged['minutos_corrida'], df_merged['valor_por_km_desc_band'])
plt.ylabel('valor por km')
plt.xlabel('minutos corrida')
plt.show()


# In[355]:


# Gráfico para avaliar a relação entre os valores de km_total, valor_corrida e minutos_corrida


# In[356]:


fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax = plt.axes(projection='3d')
xline = df_merged['km_total']
yline = df_merged['valor_corrida']
zline = df_merged['minutos_corrida']
ax.scatter3D(xline, yline, zline)
ax.set_xlabel('km_total')
ax.set_ylabel('valor_corrida')
ax.set_zlabel('minutos_corrida')


# In[357]:


# Gráfico de minutos_corrida por valor_corrida


# In[358]:


plt.scatter(df_merged['minutos_corrida'], df_merged['valor_corrida'])
plt.ylabel('valor corrida')
plt.xlabel('minutos corrida')
plt.show()


# In[359]:


# Gráfico de minutos_corrida por km_total


# In[360]:


plt.scatter(df_merged['minutos_corrida'], df_merged['km_total'])
plt.ylabel('km total')
plt.xlabel('minutos corrida')
plt.show()


# In[361]:


# Gráfico de km_total por valor_corrida


# In[362]:


plt.scatter(df_merged['km_total'], df_merged['valor_corrida'])
plt.ylabel('valor corrida')
plt.xlabel('km total')
plt.show()


# In[365]:


# Aplicação do algoritmo K-means nos dados obtidos pelas colunas dif_distancias e dif_valorKM_band1


# In[366]:


# Primeiramente, os dados serão colocados em escalas mais próximas com RobustScaler


# In[367]:


from sklearn.preprocessing import RobustScaler


# In[368]:


filter_difs_rs = df_merged[(df_merged.dif_distancias > -100) & (df_merged.dif_distancias < 200) 
                        & (df_merged.dif_valorKM_band1 < 20)]


# In[369]:


array_difs_rs = filter_difs_rs[['dif_distancias', 'dif_valorKM_band1']].to_numpy()


# In[370]:


transformer = RobustScaler().fit(array_difs_rs)


# In[371]:


transform = transformer.transform(array_difs_rs)


# In[372]:


plt.scatter(transform[:,0], transform[:,1])
plt.ylabel('dif_valorKM_band1')
plt.xlabel('dif_distancias')


# In[373]:


# Aplicação do algoritmo K-means


# In[374]:


from sklearn.cluster import KMeans


# In[375]:


# Definição do nº de clusters pelo Método do Cotovelo


# In[376]:


Sum_of_squared_distances = []
K = range(1, 10)
for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(transform)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# In[377]:


# Aplicação do K-means com 4 clusters


# In[378]:


model_k_4 = KMeans(4)
model_k_4.fit(transform)
y_model_k_4 = model_k_difs_4.predict(transform)

plt.scatter(transform[:, 0], transform[:, 1], c=y_model_k_4, s=50, cmap='viridis')

centers = model_k_4.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('dif_distancias') 
plt.ylabel('dif_valorKM_band1') 


# In[379]:


# Aplicação do K-means com 5 clusters


# In[380]:


model_k_5 = KMeans(5)
model_k_5.fit(transform)
y_model_k_5 = model_k_5.predict(transform)

plt.scatter(transform[:, 0], transform[:, 1], c=y_model_k_5, s=50, cmap='viridis')

centers = model_k_5.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('dif_distancias') 
plt.ylabel('dif_valorKM_band1') 


# In[382]:


# Aplicação do algoritmo DBSCAN 


# In[383]:


from sklearn.cluster import DBSCAN


# In[384]:


from sklearn.neighbors import NearestNeighbors


# In[385]:


neigh = NearestNeighbors(n_neighbors = 2)
nbrs = neigh.fit(transform)
distances, indices = nbrs.kneighbors(transform)


# In[386]:


distances = np.sort(distances, axis = 0)
distances = distances[:,1]
plt.plot(distances)
plt.ylim(top=0.6)


# In[387]:


model_db = DBSCAN(eps=0.2)
model_db.fit(transform)
plt.scatter(transform[:,0], transform[:,1], c=model_db.labels_)
plt.xlabel('dif_distancias') 
plt.ylabel('dif_valorKM_band1') 


# In[388]:


# Aplicação do algoritmo HDBSCAN 


# In[389]:


import hdbscan


# In[390]:


clusters_sizes = []
num_outliers = []

for k in range (2,50):
 clusterer = hdbscan.HDBSCAN(min_cluster_size=k)
 clusterer.fit(transform)
 clusters_sizes.append(len(set(clusterer.labels_)))
 num_outliers.append(sum(clusterer.labels_ == -1))


# In[391]:


plt.style.use("fivethirtyeight")
plt.figure(figsize=(12,6))
plt.plot(range(2, 50), clusters_sizes, marker='o')
plt.xticks(range(2, 50))
plt.xlabel("min_cluster_size")
plt.ylabel("Número de Clusters")
plt.xlim(0,20)
plt.show()


# In[392]:


plt.style.use("fivethirtyeight")
plt.figure(figsize=(12,6))
plt.plot(range(2, 50), num_outliers, marker='o')
plt.xticks(range(2, 50))
plt.xlabel("min_cluster_size")
plt.ylabel("Quantidade de Outliers")
plt.xlim(0,20)
plt.show()


# In[393]:


model_hdb = hdbscan.HDBSCAN(min_cluster_size=16)
model_hdb.fit(transform)
plt.scatter(transform[:,0], transform[:,1], c=model_hdb.labels_)
plt.xlabel('dif_distancias') 
plt.ylabel('dif_valorKM_band1') 


# In[394]:


# Análise dos resultados


# In[395]:


# Criar df com outliers (considerando o resultado com HDBSCAN)


# In[396]:


# Criar coluna HDBSCAN_labels


# In[397]:


df_merged['HDBSCAN_labels'] = 0.0


# In[398]:


# Os valores discrepantes que foram retirados do array na análise já são claramente outliers
# Então esses registros serão preenchidos por -1


# In[399]:


filter_outliers = df_merged[(df_merged.dif_distancias < -100) | (df_merged.dif_distancias > 200) 
                        | (df_merged.dif_valorKM_band1 > 20)]


# In[400]:


filter_outliers


# In[401]:


df_merged['HDBSCAN_labels'] = np.where((df_merged.dif_distancias < -100) | (df_merged.dif_distancias > 200) 
                        | (df_merged.dif_valorKM_band1 > 20), -(1.0), df_merged['HDBSCAN_labels'])


# In[402]:


df_merged['HDBSCAN_labels'].unique()


# In[403]:


count = df_merged[df_merged.HDBSCAN_labels == -1.0].HDBSCAN_labels.value_counts()
count


# In[404]:


filter_not_outliers = df_merged[(df_merged.dif_distancias > -100) & (df_merged.dif_distancias < 200) 
                        & (df_merged.dif_valorKM_band1 < 20)]


# In[405]:


filter_not_outliers


# In[406]:


label_idx = 0
for i in range(2837):
    if (df_merged.loc[i, 'dif_distancias'] > -100) & (df_merged.loc[i, 'dif_distancias'] < 200) & 
    (df_merged.loc[i, 'dif_valorKM_band1'] < 20):
        df_merged.loc[i, 'HDBSCAN_labels'] = model_hdb.labels_[label_idx]
        label_idx = label_idx + 1


# In[407]:


df_merged.HDBSCAN_labels.value_counts()


# In[408]:


df_outliers = df_merged[(df_merged.HDBSCAN_labels == -(1.0))]
df_outliers


# In[409]:


df_outliers.status_corrida.value_counts() 


# In[410]:


df_outliers.base_origem.value_counts()


# In[411]:


orgaos_out = df_outliers.nome_orgao.value_counts()
orgaos_out


# In[412]:


motivos_out = df_outliers.motivo_corrida.value_counts()
motivos_out


# In[413]:


# Analisar alguns gráficos feitos inicialmente, mas agora com o resultado do modelo


# In[414]:


plt.scatter(filter_difs_rs['dist_coords'], filter_difs_rs['km_total'], c=model_hdb.labels_)
plt.xlabel('dist_coords') 
plt.ylabel('km_total') 
# pontos fora da reta como outliers


# In[415]:


# Análise dos gráficos abaixo:
# Pontos muito discrepantes foram realmente detectados como outliers. Mas também mostra que pontos dentro da reta, 
# que à primeira vista não seriam vistos como outliers, podem sim apresentar alguma inconformidade e devem ser verificados.


# In[416]:


plt.scatter(df_merged['minutos_corrida'], df_merged['valor_corrida'], c=df_merged['HDBSCAN_labels'])
plt.xlabel('minutos_corrida') 
plt.ylabel('valor_corrida') 


# In[417]:


plt.scatter(df_merged['minutos_corrida'], df_merged['km_total'], c=df_merged['HDBSCAN_labels'])
plt.xlabel('minutos_corrida') 
plt.ylabel('km_total') 


# In[418]:


plt.scatter(df_merged['km_total'], df_merged['valor_corrida'], c=df_merged['HDBSCAN_labels'])
plt.xlabel('km_total') 
plt.ylabel('valor_corrida') 


# In[266]:


# Exportar df_merged para excel


# In[267]:


import openpyxl


# In[268]:


df_merged.to_excel('df_merged_final.xlsx', index=False)


# In[ ]:




