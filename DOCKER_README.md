# Docker Setup for RichardsDrive Car Defect Detection

This guide explains how to containerize and run the RichardsDrive backend using Docker.

## Prerequisites

- Docker Desktop installed and running
- At least 4GB of available RAM
- 10GB of free disk space

## Quick Start

### 1. Environment Setup

Copy the environment template:
```bash
cp .env.example .env
```

Edit `.env` file and add your API keys if needed:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the backend service
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 3. Verify the Service

Once running, verify the service:
- Health check: http://localhost:8000/health
- API documentation: http://localhost:8000/docs
- Real-time detection: ws://localhost:8000/ws/realtime-detection

## Manual Docker Commands

### Build the Image
```bash
cd backend
docker build -t richards-drive-backend .
```

### Run the Container
```bash
docker run -d \
  --name richards-drive-backend \
  -p 8000:8000 \
  -v richards_temp:/app/temp \
  -v richards_uploads:/app/uploads \
  --env-file ../.env \
  richards-drive-backend
```

## Container Management

### View Logs
```bash
# Docker Compose
docker-compose logs -f backend

# Direct Docker
docker logs -f richards-drive-backend
```

### Stop Services
```bash
# Docker Compose
docker-compose down

# Direct Docker
docker stop richards-drive-backend
docker rm richards-drive-backend
```

### Restart Services
```bash
# Docker Compose
docker-compose restart backend

# Direct Docker
docker restart richards-drive-backend
```

## Production Deployment

### Resource Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB for models and temporary files

### Security Considerations
1. Use specific CORS origins instead of `*`
2. Set up proper SSL/TLS certificates
3. Use secrets management for API keys
4. Enable Docker security scanning

### Scaling
For high-traffic scenarios, consider:
- Running multiple backend instances
- Using a load balancer (nginx)
- Implementing Redis for session management

## Troubleshooting

### Common Issues

1. **Docker not running**
   ```
   ERROR: error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping"
   ```
   Solution: Start Docker Desktop

2. **Out of memory**
   ```
   Container killed due to memory limit
   ```
   Solution: Increase Docker memory allocation or reduce model batch size

3. **Port already in use**
   ```
   Port 8000 is already allocated
   ```
   Solution: Change port mapping in docker-compose.yml or stop conflicting service

4. **Model files not found**
   ```
   FileNotFoundError: No training runs found
   ```
   Solution: Ensure model files (*.pt) are present in the backend directory

### Health Check
The container includes a health check endpoint at `/health`. If the container shows as unhealthy:
1. Check logs for errors
2. Verify all model files are present
3. Ensure sufficient memory allocation

## Development

### Hot Reload Development
For development with hot reload:
```bash
docker run -d \
  --name richards-dev \
  -p 8000:8000 \
  -v $(pwd)/backend:/app \
  -e PYTHONPATH=/app \
  python:3.11-slim \
  bash -c "cd /app && pip install -r requirements_api.txt && python app.py"
```

### Debugging
To debug inside the container:
```bash
docker exec -it richards-drive-backend bash
```

## API Endpoints

Once running, the following endpoints are available:

- `GET /health` - Health check
- `POST /analyze` - Single image analysis
- `POST /analyze-all` - Multi-model analysis
- `WS /ws/realtime-detection` - Real-time detection WebSocket
- `GET /docs` - Interactive API documentation

## File Structure

```
backend/
├── Dockerfile              # Container definition
├── .dockerignore           # Files to exclude from build
├── requirements_api.txt    # API dependencies
├── app.py                 # Main application
├── inference.py           # Inference logic
├── ensemble_logic.py      # Ensemble methods
├── model2/               # Model 2 files
├── model3/               # Model 3 files
├── model4/               # Model 4 files
└── *.pt                  # Model weights
```