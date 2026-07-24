FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY infra_advisor.py advisor_extensions.py domains.py compliance.py arb_report.py blueprints.py ./
COPY data/ ./data/

ENV PORT=8080
EXPOSE 8080

# Multi-domain app is the default entrypoint; storage-only app kept for reference
CMD streamlit run infra_advisor.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
