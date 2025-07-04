FROM python:3.13-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* ./

COPY . .
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root && \
    poetry build

EXPOSE 8000

CMD ["poetry", "run", "fastapi", "run", "src/server.py"]