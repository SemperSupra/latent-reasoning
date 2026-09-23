FROM python:3.12-slim

ARG SOURCE_REVISION=UNPINNED
ENV LATENT_REASONING_SOURCE_REVISION=${SOURCE_REVISION}
ENV LATENT_REASONING_CONTAINER=cpu-replay-v1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements-experiment-cpu.txt ./
RUN python -m pip install --no-cache-dir -r requirements-experiment-cpu.txt

COPY experiments ./experiments
COPY tools ./tools
COPY replays ./replays
COPY environments ./environments

ENTRYPOINT ["python", "tools/run_replay.py"]
