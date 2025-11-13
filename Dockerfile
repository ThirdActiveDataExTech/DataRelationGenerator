FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder-base

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY uv.lock pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --link-mode=copy


# `production` image used for runtime
FROM python:3.11-slim AS production

# Setting home directory and user name
ENV APP_HOME=/home/wisenut/app \
    GROUP_NAME=wisenut \
    APP_USER=wisenut

# Create a non-root user and group
RUN groupadd -r $GROUP_NAME && useradd -r -g $GROUP_NAME -d $APP_HOME $APP_USER

# Set the working directory
WORKDIR $APP_HOME

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy necessary files and directory (as root)
COPY --from=builder-base $VIRTUAL_ENV $VIRTUAL_ENV
COPY pyproject.toml .env ./
COPY ./static ./static/
COPY ./app ./app/

# Generate version info file with proper environment (as root)
RUN python app/version.py

# Set ownership of all files to the app user
RUN chown -R $APP_USER:$GROUP_NAME $APP_HOME

# Switch to the non-root user
USER $APP_USER

# Expose the port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]