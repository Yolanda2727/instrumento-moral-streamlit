# ETHOSCOPE

ETHOSCOPE es una plataforma académica en Streamlit orientada al análisis formativo del razonamiento moral, la deliberación ética y la calidad básica de la argumentación frente a dilemas contextualizados.

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
- Profesiones individuales en el selector: Medicina, Enfermería, Fisioterapia, Instrumentación Quirúrgica, Bacteriología, Microbiología, Derecho, Ciencias Sociales, Educación, Ingeniería, TI, Datos y Otra / Mixta.
- Rutas de dilemas compatibles con el esquema anterior, ahora asignadas automáticamente según la profesión seleccionada.
- Subrutas específicas para Bacteriología y Microbiología, con dilemas diferenciados sobre bioseguridad, transporte de muestras, contaminación cruzada, resistencia antimicrobiana, reporte crítico, cultivos, secuenciación e investigación con muestras biológicas.
- Vista administrativa para revisar y exportar el catálogo de dilemas de laboratorio con foco temático y justificación pedagógica.
- Reporte individual inmediato.
- Dashboard colectivo con distribuciones, brechas y análisis cualitativo exploratorio.
- Nueva sección de Interpretación IA con lectura individual y grupal apoyada por OpenAI.
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

La app usa esta estrategia de persistencia:

- Si existe `SUPABASE_DB_URL`, usa **Supabase/Postgres** como backend principal.
- Si no existe, usa **SQLite local** como fallback de desarrollo.
- Si hay un CSV legado en `data/responses.csv`, puede migrarlo automáticamente al backend activo en el primer arranque.

Por defecto, el fallback local guarda respuestas en:

```text
data/responses.db
```

Ruta de CSV legado compatible:

```text
data/responses.csv
```

Puedes cambiar la ruta del backend SQLite o del CSV legado con variables de entorno:

```bash
export MORAL_TEST_SQLITE_PATH="/ruta/personalizada/responses.db"
export MORAL_TEST_LEGACY_CSV_PATH="/ruta/personalizada/responses.csv"
```

Para implementación institucional real, configura:

```bash
export SUPABASE_DB_URL="postgresql://usuario:password@host:5432/postgres"
```

En **Streamlit Community Cloud**, el disco local sigue siendo efímero. Para uso académico serio, la opción recomendada es Supabase/Postgres.

## Reportes administrativos

La app guarda reportes automáticos en una carpeta administrativa del servidor:

```text
data/admin_reports
```

Incluye:

- Reportes individuales en JSON y CSV por intento.
- Snapshots colectivos globales y filtrados para revisión administrativa.

Puedes cambiar la ruta con:

```bash
export MORAL_TEST_ADMIN_REPORTS_DIR="/ruta/privada/admin_reports"
```

Si quieres que esos archivos queden en Google Drive, la carpeta de Drive debe estar sincronizada o montada localmente en el servidor. Luego apuntas `MORAL_TEST_ADMIN_REPORTS_DIR` a esa ruta local. Un enlace web de Drive, por sí solo, no permite escritura directa desde la app.

## Acceso administrador

Las vistas:

- Dashboard colectivo
- Administración

quedan protegidas con una contraseña de administrador.

Configúrala así:

```bash
export MORAL_TEST_ADMIN_PASSWORD="tu-clave-segura"
```

También puedes definirla en `st.secrets` si despliegas en Streamlit Community Cloud.

La interpretación IA requiere además:

```toml
OPENAI_API_KEY = "tu-clave-openai"
OPENAI_MODEL = "gpt-4.1-mini"
```

## Salida JSON estructurada de Interpretación IA

La sección Interpretación IA usa Structured Outputs de OpenAI para devolver una respuesta JSON estable y renderizable directamente en Streamlit.

Estructura esperada:

```json
{
    "resumen_ejecutivo": "",
    "interpretacion_etica": "",
    "interpretacion_legal": "",
    "interpretacion_bioetica": "",
    "riesgos": [
        {
            "riesgo": "",
            "descripcion": "",
            "nivel": ""
        }
    ],
    "consideraciones_clave": [""],
    "fortalezas_argumentativas": [""],
    "debilidades_argumentativas": [""],
    "recomendaciones_formativas": [""],
    "tabla_analitica": [
        {
            "dimension": "",
            "hallazgo_principal": "",
            "interpretacion": "",
            "riesgo_asociado": "",
            "nivel_atencion": "",
            "recomendacion": ""
        }
    ]
}
```

Uso dentro de la app:

- `resumen_ejecutivo`, `interpretacion_etica`, `interpretacion_legal` e `interpretacion_bioetica` se muestran como bloques narrativos.
- `riesgos` se renderiza como tarjetas con nombre del riesgo, descripción y nivel.
- `consideraciones_clave`, `fortalezas_argumentativas`, `debilidades_argumentativas` y `recomendaciones_formativas` se renderizan como listas.
- `tabla_analitica` se renderiza como tabla final para lectura comparativa y exportación.

El esquema estricto se define en [services/openai_interpreter.py](services/openai_interpreter.py) y el renderizado se implementa en [utils/render_interpretation.py](utils/render_interpretation.py).

Para desarrollo local ya puedes usar el archivo:

```text
.streamlit/secrets.toml
```

con este valor mínimo:

```toml
MORAL_TEST_ADMIN_PASSWORD = "cambia-esto-por-una-clave-segura"
OPENAI_API_KEY = "tu-clave-openai"
OPENAI_MODEL = "gpt-4.1-mini"
```

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
