# Backend Docker Deployment

This directory contains Docker configuration files to containerize and deploy the Richards Drive backend application.

## Files Created

- `Dockerfile` - Docker image configuration
- `.dockerignore` - Files to exclude from Docker build context
- `docker-compose.yml` - Docker Compose configuration for easy deployment

## Prerequisites

- Docker installed on your system
- Docker Compose installed (usually comes with Docker Desktop)
- Model weights should be available in the `../runs/` directory

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   cd backend
   docker-compose up --build
   ```

2. **Run in detached mode:**
   ```bash
   docker-compose up -d --build
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker Commands

1. **Build the Docker image:**
   ```bash
   cd backend
   docker build -t richardsdrive-backend .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -v "$(pwd)/../runs:/app/../runs:ro" \
     -e GEMINI_API_KEY="your_api_key_here" \
     richardsdrive-backend
   ```

## Configuration

### Environment Variables

- `GEMINI_API_KEY` - Your Gemini API key (optional)
- `PYTHONPATH` - Set to `/app` (automatically configured)
- `PYTHONUNBUFFERED` - Set to `1` for real-time logs (automatically configured)

### Ports

- The application runs on port `8000` inside the container
- Port `8000` is exposed and mapped to host port `8000`

### Volumes

- `../runs:/app/../runs:ro` - Mounts model weights directory as read-only
- For development, you can uncomment the volume mount in docker-compose.yml to mount the source code

## Health Check

The container includes a health check that verifies the application is running properly:
- Endpoint: `http://localhost:8000/health`
- Check interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds

## Accessing the Application

Once the container is running, you can access:
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- WebSocket endpoint: ws://localhost:8000/ws/realtime-detection

## Troubleshooting

### View container logs:
```bash
docker-compose logs -f backend
```

### Access container shell:
```bash
docker-compose exec backend bash
```

### Rebuild without cache:
```bash
docker-compose build --no-cache
```

### Check container status:
```bash
docker-compose ps
```

## Production Considerations

1. **Security**: Set proper environment variables and secrets
2. **Scaling**: Use Docker Swarm or Kubernetes for production scaling
3. **Monitoring**: Add proper logging and monitoring solutions
4. **SSL/TLS**: Configure reverse proxy (nginx) for HTTPS
5. **Resource Limits**: Set memory and CPU limits in docker-compose.yml

## Model Weights

Ensure your model weights are available in the `../runs/` directory structure:
```
runs/
├── train/
│   ├── yolov8s/
│   │   └── weights/
│   ├── model2/
│   │   └── weights/
│   ├── model3/
│   │   └── weights/
│   └── model4/
│       └── weights/
```

The container will create these directories if they don't exist, but you need to ensure the actual model weight files are present.