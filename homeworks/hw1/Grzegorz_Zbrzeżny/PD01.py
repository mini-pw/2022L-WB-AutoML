#!/usr/bin/env python
# coding: utf-8

# # Preprocessing danych

# In[262]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import re
import math


# In[263]:


data = pd.read_csv("diabetic_data.csv")
data.head()


# # Identyfikacja zmiennej objaśnianej

# Za pomocą tego data setu chcemy przewidzieć, czy pacjent powróci do szpitala po zwolnieniu go z niego (etykieta `readmitted`). Możemy więc od razu wykluczyć pewne kolumny, które to na pewno nie są z tym związane. 
# * `encounter_id` - to unikalna wartość dla każdego przyjęia do szpitala, nie ma związku z jakimikolwiek czynnikami chorobowymi, można więc usunąć te kolumnę, by uniknąć potencjalnego przeuczenia
# * `patient_nbr` - jest indywidualna dla każdego pacjenta, więc zostawienie jej ponownie mogłoby prowadzić do przeuczenia
# 
# 
# Pozostałe kolumny są już mniej lub bardziej związane z czynnikami chorobowymi (przyjmowane substancje leczące, rodzaj diagnozy, itd.), więc mogą decydować o readmisji danego pacjenta. Dokładny opis poszczególnych kolumn można znaleźć pod linkiem: https://www.hindawi.com/journals/bmri/2014/781670/tab1/ dlatego nie będę już przepisywał tych opisów do raportu

# # Typy kolumn

# In[264]:


data.info()


# # Identyfikacja brakujących danych i imputacja

# In[265]:


data.drop('encounter_id', axis = 1, inplace = True)
data.drop('patient_nbr', axis = 1, inplace = True)


# Sprawdźmy, czy któreś inne kolumny nie powinny zostać usunięte z jakichś powodów

# In[266]:


per = data["weight"][data["weight"] == "?"].count()/data["weight"].count()*100
print(f"Ilość '?' w kolumnie weight: {per:.{2}f}%")


# Widać stąd, że kolumnę `weight` można bez problemu usunąć, gdyż braki danych "?" stanowią aż ponad 96% wszystkich danych

# In[267]:


data.drop('weight', axis = 1, inplace = True)


# In[268]:


print(data["medical_specialty"][data["medical_specialty"] == "?"].count()/data["medical_specialty"].count())
data.drop('medical_specialty', axis = 1, inplace = True)


# Usuwam kolummnę `medical_speciality`, ponieważ ma aż prawie 50% braków.

# In[269]:


print(data["payer_code"][data["payer_code"] == "?"].count()/data["payer_code"].count())
data.drop('payer_code', axis = 1, inplace = True)


# Usuwam kolummnę `payer_code`, bo kod nie ma zbyt wiele wspólnego ze zmienną `readmitted`, a braki danych stanowią aż prawie 40%

# Pod linkiem wymienionym w Identyfikacji zmiennej objaśnianej możemy zobaczyć, że w kolumnach `Race` i `diag_3` jest odpowiednio 2% i 1% braków danych. Jako, że to naprawdę niewiele danych możemy po prostu usunąć te wiersze, w których wystąpi brak danych w tych kolumnach. To samo w kolumnach `diag_1` i `diag_2`, w których również występują braki danych w postaci "?" i również jest ich naprawdę mało.

# In[270]:


data = data.drop(data[data.race == "?"].index)
data = data.drop(data[data.diag_3 == "?"].index)
data = data.drop(data[data.diag_1 == "?"].index)
data = data.drop(data[data.diag_2 == "?"].index)


# Ponadto usuńmy rekordy, w których płeć ma wartość inną niż `Female`/`Male`, ponieważ jest ich bardzo mało, a konkretnie

# In[271]:


data["gender"].value_counts()


# In[272]:


data = data.drop(data[data.gender == "Unknown/Invalid"].index)


# Możemy usunąć też kolummy o stałych wartościach. Z heatmapy, która jest w sekcji eksploracji danych widać, że kolumny o stałych wartościach to `citoglipton`, `examide` i `metformin-rosiglitazone`.

# In[273]:


data.drop(["citoglipton", "examide", "metformin-rosiglitazone"], axis = 1, inplace = True)


# # Encoding zmiennych kategorycznych

# Wpierw możemy encodować kolumny z substancjami medycznymi przyjmowanymi przez pacjentów.

# In[274]:


le = preprocessing.LabelEncoder()
le.fit(data["insulin"])
print(f"Wartości w tych kolumnach to: {list(le.classes_)}")


# Jako, że w tych kolumnach występują jedynie 4 różne wartości, w tym `No` oznacza, że pacjent w ogóle nie przyjmuje danej substancji, natomiast pozostałe wartości to różne warianty przyjmowania (`Down` - przyjmuje mniej, `Steady` - przyjmuje tyle samo, `Up` - przyjmuje więcej). Możemy więc zastosować kodowanie, gdzie 0 będzie sytuacja, gdy pacjent w ogóle nie przyjmuje leku, a 1 gdy przyjmuje.

# In[275]:


dict = {"No" : 0, "Down" : 1, "Steady" : 1, "Up" : 1}
for i in range(19, 39):
    data.iloc[:, i] = data.iloc[:, i].map(dict)


# Ponadto możemy to również zrobić dla kolumn `gender`, `change`, oraz `diabetesMed`, gdyż są to kolumny binarne.

# In[276]:


names = ["gender", "change", "diabetesMed"]
for i in names:
    le2 = preprocessing.LabelEncoder()
    le2.fit(data[i])
    data[i] = le2.transform(data[i])


# Dla kolumn `race`, `max_glu_serum` i `A1Cresult` wykonamy One Hot Encoding, gdyż przyjmują one 4, bądź 5 różnych wartości. Dodatkowo możemy zrobić one hot'a dla kolumny `admission_type_id` i `num_procedures`, gdyż zawierają one tylko 7 różnych wartości.

# In[277]:


names = ["race", "A1Cresult", "admission_type_id", "num_procedures", "max_glu_serum"]
for i in names:
    cols = pd.get_dummies(data[i], prefix=i)
    data.drop(i, axis = 1, inplace = True)
    data = data.join(cols)


# In[278]:


data["readmitted"].unique()


# Ponadto użyjmy label encodingu na etykiecie, gdyż jak widać przyjmuje ona 4 wartości nieliczbowe.

# In[279]:


dict = {"NO" : 0, "<30" : 1, ">30" : 2}
data["readmitted"] = data["readmitted"].map(dict)


# # Transformacje danych

# Możemy przetransformować niektóre kolumny zawierające wiele wartości liczbowych poprzez zgrupowanie tych wartości przedziałami. Zrobimy to po kolei na kolumnach `time_in_hospital`, `number_outpatient`

# Nie potrzeba nam raczej informacji o dokładnej ilości dni tylko wystarczy, aby to była ogólna informacja o długości pobytu, pogrupujmy więc wartości od 0 do 14 na grupy `short - 0`, `medium - 1` i `long - 2`.

# In[280]:


data["time_in_hospital"] = pd.cut(data["time_in_hospital"], bins = [0., 4, 9, np.inf], labels = [0, 1, 2])


# Dla kolumn `number_outpatient`, `number_emergency` i `number_inpatient` wartości powyżej 10 występują po naprawdę niewiele razy w stosunku do tych od 0 do 10, więc możemy zastąpić je wartością 10.

# In[281]:


data.loc[data["number_outpatient"] >= 10, "number_outpatient"] = 10
data.loc[data["number_emergency"] >= 10, "number_emergency"] = 10
data.loc[data["number_inpatient"] >= 10, "number_inpatient"] =10


# Przekształćmy kolumnę age na wartości numeryczne. Skoro mamy podane przedziały wieku możemy bez straty ogólności przekształcić je na środki tych przedziałów

# In[282]:


def extract_and_get_mean(x):
    values = re.findall(r'\d+', x)
    return (int(values[0]) + int(values[1]))/2


# In[283]:


data["age"] = data["age"].apply(extract_and_get_mean)


# In[284]:


data = data.astype({"time_in_hospital" : int, "age" : int})


# Dodatkowo możemy pogrupować wartości diagnoz z grupy pierwszej (diagnoza główna) według ich grup omówionych pod linkiem: https://www.hindawi.com/journals/bmri/2014/781670/tab2/ . Natomiast kolumny `diag_2` i `diag_3` usunę, gdyż są to diagnozy dodatkowe, tak więc najważniejsze wartości zawiera kolumna `diag_1`.

# In[285]:


alfabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# In[286]:


def group_diagnoses(x):
    if (x[0].lower() in alfabet or (280 < math.floor(float(x)) < 289) or (320 < math.floor(float(x)) < 329) or (630 < math.floor(float(x)) < 679) or (360 < math.floor(float(x)) < 389) or (740 < math.floor(float(x)) < 749)):
        return "Other"
    elif (390 < math.floor(float(x)) < 459 or math.floor(float(x)) == 785):
        return  "Circulatory"
    elif (460 < math.floor(float(x)) < 519 or math.floor(float(x)) == 786):
        return  "Respiratory"
    elif (520 < math.floor(float(x)) < 579 or math.floor(float(x)) == 787):
        return "Digestive"
    elif (800 < math.floor(float(x)) < 999):
        return "Injury"
    elif (710 < math.floor(float(x)) < 739):
        return "Musculoskeletal"
    elif (580 < math.floor(float(x)) < 629 or math.floor(float(x)) == 788):
        return  "Genitourinary"
    elif (math.floor(float(x)) == 250):
        return  "Diabetes"
    else:
        return "Neoplasms"


# In[287]:


data["diag_1"] = data["diag_1"].apply(group_diagnoses)


# In[291]:


data.drop('diag_2', axis = 1, inplace = True)
data.drop('diag_3', axis = 1, inplace = True)


# Na koniec zastosujmy One hot encoding do kolumny `diag_1`, gdyż zawiera ona tylko 9 unikalnych wartości.

# In[293]:


cols = pd.get_dummies(data["diag_1"], prefix="diag_1")
data.drop("diag_1", axis = 1, inplace = True)
data = data.join(cols)


# In[294]:


data["diag_1"].value_counts()


# # Eksploracyjna analiza danych

# Sprawdźmy ponownie typy danych

# In[295]:


data.info()


# ## Analiza jednowymiarowa

# In[296]:


data.hist(bins= 30,figsize = (28, 23))
plt.show()


# Stwórzmy `describe` dla danych przyjmujących więcej niż 5 różnych wartości gdyż tylko wtedy ma to większy sens

# In[301]:


data.iloc[:, [1, 5, 6, 8, 9, 10]].describe()


# ### Wnioski z analizy jednowymiarowej
# * większość zmiennych przyjmuje bardzo ograniczoną ilość wartości
# * większość pacjentów w ogóle nie przyjmuje leków wymienionych w tym secie danych
# * kolumny `age`, `num_lab_procedures` i `num_medications` dane są rozkładem poodobnym do normalnego (skośny), najprawdopodobniej mozna by je przekształcić do rozkładu o wiele bardziej podobnego do normalnego za pomocą logarytmowania

# ## Analiza wielowymiarowa

# Stworzymy macierz korelacji, ale na danych sprzed one hot encodingu

# In[297]:


matrix = data.iloc[:, :34].corr()
plt.figure(figsize=(20, 16))
sns.heatmap(matrix, cmap = 'coolwarm', center = .0)

plt.show()


# In[298]:


x = pd.DataFrame(data.groupby(by = ["diabetesMed", "gender"], as_index=False)["change"].agg("sum"))
sns.barplot(data = x, x = "diabetesMed", y = "change", hue = "gender")
plt.show()


# Widać, że w przypadku pacjentów, u których zastosowano leki dla cukrzyków (w tym przypadku: `1 - zastosowano`, `0 - nie zastosowano`) częściej nie było zmiany w lekach cukrzycowych (w tym przypadku `0 - zmiana`, `1 - jej brak`). Ponadto widać, że płeć nie miała tu zbyt dużego znaczenia, gdyż w obydwu przypadkach stosunek ilości pacjentów różnej płci, u których nie potrzeba było zmieniać leków jest podobny.

# In[299]:


x = pd.DataFrame(data.groupby(by = ["time_in_hospital", "gender"], as_index=False)["num_medications"].agg("mean"))
sns.barplot(data = x, x = "time_in_hospital", y = "num_medications", hue = "gender")
plt.show()


# Z wykresu można wywnioskować, że często razem z długością pobytu w szpitalu, rośnie też ilość przyjętych przez pacjenta rodzajów leków, co wydaje się dość logiczne. Widać też, że i tym razem płeć nie ma wielkiego znaczenia przy którejkolwiek z tych danych.

# Przypomnijmy oznaczenia zmiennej `time_in_hospital`:
# * `0 - short`
# * `1 - medium`
# * `2 - long`

# In[300]:


sns.boxplot(data = data, x = "time_in_hospital", y = "num_lab_procedures")
plt.show()


# Widać, że wraz ze wzrostem długości pobytu w szpitalu, rosła też ilość szpitalnych procedur z wyłączeniem testów laboratoryjnych, co również wydaje się bardzo logiczne
