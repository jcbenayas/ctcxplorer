# CTCXplorer 🚦

Analizador interactivo de ficheros de log de CTC (Control de Tráfico Centralizado)  
con **DuckDB**, **Pandas** y **Gradio**.

## Funcionalidades

| Pestaña      | Descripción |
|--------------|-------------|
| **Trenes**   | Selecciona circuitos y visualiza el momento en que cada tren los atraviesa. |
| **Velocidades** | Elige dos circuitos + distancias y calcula la velocidad media de cada tren entre ellos. |
| **Mandos**   | Filtra mensajes `#M` con búsquedas simples (`AND`, `OR`, `NOT`). |

## Instalación local

```bash
# 1. Clona el repo
git clone https://github.com/tu-usuario/ctcxplorer.git
cd ctcxplorer

# 2. Crea entorno y actívalo
python -m venv .venv
source .venv/bin/activate

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Lanza la app
python ctcxplorer.py
# Abre http://127.0.0.1:7860 en tu navegador
