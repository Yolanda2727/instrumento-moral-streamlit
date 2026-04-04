# Instrumento formativo de razonamiento moral y marcos éticos

Aplicación en Streamlit para estudiantes y profesionales orientada al análisis formativo del razonamiento moral, la deliberación ética y la calidad básica de la argumentación frente a dilemas contextualizados.

## Autor

**Profesor Anderson Díaz Pérez**  
- Doctor en Bioética  
- Doctor en Salud Pública  
- Magíster en Ciencias Básicas Biomédicas  
- Especialista en Inteligencia Artificial  
- Profesional en Instrumentación Quirúrgica  

## Función principal

Valorar de manera formativa cómo estudiantes y profesionales argumentan frente a dilemas éticos, integrando razonamiento moral, marcos éticos y calidad básica de la justificación escrita.

## Objetivos del programa

1. Fortalecer la deliberación ética mediante dilemas contextualizados por área profesional.
2. Identificar patrones argumentativos asociados a estadios morales y marcos éticos dominantes.
3. Generar retroalimentación individual inmediata para docencia, reflexión y mejora argumentativa.
4. Consolidar visualizaciones colectivas útiles para análisis académico, formación ética e investigación educativa exploratoria.

## Qué incluye esta versión

- Portada institucional dentro de la app.
- Identificación visible del autor y sus credenciales.
- Presentación gráfica del flujo funcional del programa.
- Rutas por profesión: salud, derecho/ciencias sociales/educación, ingeniería/TI/datos y mixta.
- Reporte individual inmediato.
- Dashboard colectivo con distribuciones, brechas y análisis cualitativo exploratorio.
- Exportación en CSV para análisis adicional.

## Alcance

- **Uso recomendado:** docencia, formación ética, simulación, reflexión académica, análisis exploratorio.
- **No usar como diagnóstico psicológico, veredicto moral definitivo ni criterio único de evaluación profesional o institucional.**

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura del proyecto

```text
moral_test_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    └── .gitkeep
```

## Persistencia

Por defecto, la app guarda respuestas en:

```text
data/responses.csv
```

Puedes cambiar la ruta con una variable de entorno:

```bash
export MORAL_TEST_DATA_PATH="/ruta/personalizada/responses.csv"
```

En **Streamlit Community Cloud**, el disco es efímero. Para implementación institucional real, se recomienda migrar el almacenamiento a una base persistente.

## Despliegue en Streamlit Community Cloud

1. Sube este proyecto a GitHub.
2. Entra a Streamlit Community Cloud.
3. Conecta tu cuenta de GitHub.
4. Crea una nueva app.
5. Selecciona el repositorio y el archivo `app.py`.
6. Haz clic en **Deploy**.

## Recomendación institucional

Para una versión robusta en producción, sustituye el CSV por:
- Supabase / PostgreSQL
- Google Sheets API
- S3 / almacenamiento objeto
- Base de datos institucional
