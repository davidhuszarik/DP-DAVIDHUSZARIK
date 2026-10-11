# Použitie oficiálneho a odľahčeného Python obrazu
FROM python:3.11-slim

# Nastavenie premenných prostredia pre Python a Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Inštalácia závislostí v samostatnej vrstve pre efektívne cachovanie
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Vytvorenie neprivilegovaného používateľa pre zvýšenie bezpečnosti
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Kopírovanie zdrojových kódov aplikácie
COPY . .
RUN chown -R appuser:appuser /app

# Prepnutie na neprivilegovaného používateľa
USER appuser

# Exponovanie portu pre Streamlit
EXPOSE 8501

# Kontrola stavu bežiaceho kontajnera (Healthcheck)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Spustenie aplikácie
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
