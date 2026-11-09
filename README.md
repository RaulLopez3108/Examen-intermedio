# Examen-intermedio
<pre>
Nombre: Raul Leonardo Lopez Ruiz | Expediente: 218207462 | a = 2, b = 6

Examen intermedio/
├── Problema 1/
│   ├── codigo_p1.py    # Script principal del Problema 1
│   └── README.md       # Documentación del Problema 1
└── Problema 2/
    ├── codigo.py       # Script principal del Problema 2
    └── README.md       # Documentación del Problema 2
</pre>

    
##### Pasos para ejecutar el problema 1
cd "C:\Users\Usuario\Desktop\Examen intermedio\Problema 1"

###### Crear y activar el entorno virtual con uv
uv venv
 \.venv\Scripts\Activate.ps1
###### Instalar dependencias científicas para clustering y reducción de dimensionalidad
uv pip install numpy pandas scikit-learn scipy matplotlib seaborn
uv codigo.py



##### Pasos para ejecutar el problema 2
 cd "C:\Users\Usuario\Desktop\Examen intermedio\Problema 2"
###### Inicializar entorno con uv 
    uv venv
     \.venv\Scripts\Activate.ps1
###### Instalar las librerías requeridas para el análisis, modelos y gráficas
 uv pip install numpy pandas scikit-learn xgboost matplotlib requests
 uv run codigo.py
 
