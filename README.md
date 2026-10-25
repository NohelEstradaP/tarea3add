# Tarea 3: Introducción a Series de Tiempo

Análisis y Ciencia de Datos, Facultad de Ingeniería, Universidad del Istmo de Guatemala.
Didvin Nohel Estrada Pineda, carné 14092, Ingeniería en Sistemas.

El informe con las derivaciones, el análisis, la bitácora de limpieza, las tablas y las figuras está en `informe/informe.pdf` (fuente en `informe/informe.tex`).

## Cómo correrlo

1. Descargar `Online Retail.xlsx` del repositorio de UCI (https://archive.ics.uci.edu/dataset/352/online+retail) y ponerlo en `data/`. No está en el repo porque pesa unos 23 MB.
2. Instalar dependencias (Python 3.10 o más reciente):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Correr todo de principio a fin (alrededor de un minuto):

   ```bash
   python src/main.py
   ```

   Esto regenera todas las tablas (`tablas/`, en `.csv` y `.tex`) y figuras (`figuras/`) que usa el informe. Cada parte también se puede correr sola, por ejemplo `python src/parte5_pronostico.py`.

4. Para compilar el informe: `cd informe && tectonic informe.tex` (o `pdflatex informe.tex`).

## Archivos

| Archivo | Contenido |
|---|---|
| `src/limpieza.py` | `limpiar(df)`: todas las reglas de limpieza en orden, con bitácora opcional |
| `src/utils.py` | rutas, ACF a mano, Ljung-Box, serie diaria y partición entrenamiento/prueba |
| `src/parte0_simulacion.py` | Parte 0: AR(1), caminata aleatoria y procesos sin memoria |
| `src/parte1_limpieza.py` | Parte 1: conteos, casos, bitácora y las seis líneas más grandes |
| `src/parte2_calendario.py` | Parte 2: serie diaria y días sin transacciones |
| `src/parte3_descomp.py` | Parte 3: media móvil, índices estacionales y ACF |
| `src/parte4_sarima.py` | Parte 4: regresión con días, AR(1), SARIMA S1 y S2 |
| `src/parte5_pronostico.py` | Parte 5: B1, B2, M1, M2, S1, S2 con tres esquemas de validación |
| `src/parte6_rfm.py` | Parte 6: RFM, AUC, campeones y clientes nuevos |
| `src/main.py` | corre todas las partes en orden |

## Semillas

- Parte 0: `default_rng(14092 + i)` para cada escenario (cada uno con semilla distinta).
- Parte 5: semillas 0 a 9 para los folds aleatorios (`KFold(shuffle=True, random_state=s)`).
- Parte 6: `default_rng(2026)` para desempatar los quintiles.
