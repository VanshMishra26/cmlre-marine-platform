# Docker Configuration for CMLRE Marine Platform

This folder contains Docker configuration files for the CMLRE Marine Platform.

## Frontend Dockerfiles

### 1. `Dockerfile` (Main)
The main Dockerfile for building the frontend. This is designed to be run from the docker folder with the frontend as build context.

**Usage:**
```bash
# Build from docker folder
docker build -f docker/Dockerfile -t cmlre-frontend ../frontend

# Or use with docker-compose (already configured)
docker-compose up frontend
```

### 2. `Dockerfile.frontend` (Alternative)
Alternative Dockerfile that references the frontend folder relatively.

**Usage:**
```bash
# Build from docker folder
docker build -f docker/Dockerfile.frontend -t cmlre-frontend .
```

### 3. `Dockerfile.frontend.standalone` (Self-contained)
A standalone version that includes all necessary configurations inline.

**Usage:**
```bash
# Copy frontend files to docker folder first, then build
cp -r ../frontend/* .
docker build -f Dockerfile.frontend.standalone -t cmlre-frontend .
```

## Backend Dockerfiles

### 1. `Dockerfile.backend` (Eclipse Temurin JDK)
Uses Eclipse Temurin JDK 17 (recommended for production) with Alpine Linux for smaller image size.

**Usage:**
```bash
# Build from docker folder
docker build -f docker/Dockerfile.backend -t cmlre-backend .

# Or use with docker-compose
docker-compose up backend
```

### 2. `Dockerfile.backend.openjdk` (OpenJDK)
Uses official OpenJDK 17 images with Alpine Linux.

**Usage:**
```bash
# Build from docker folder
docker build -f docker/Dockerfile.backend.openjdk -t cmlre-backend .
```

### 3. Updated Backend Dockerfile
The main backend Dockerfile has been updated to use Eclipse Temurin JDK 17 with Alpine Linux for better performance and security.

## Features

### Frontend Dockerfiles include:
- **Multi-stage build** for optimized production images
- **Node.js 18 Alpine** for the build stage
- **Nginx Alpine** for the production stage
- **Security headers** and optimizations
- **Health checks** for container monitoring
- **Non-root user** for security
- **Gzip compression** for better performance
- **API proxying** to backend and ML services
- **Client-side routing** support for React

### Backend Dockerfiles include:
- **Eclipse Temurin JDK 17** (recommended) or **OpenJDK 17**
- **Alpine Linux** for smaller image size and better security
- **Multi-stage build** with separate build and runtime stages
- **Maven** for dependency management and building
- **Non-root user** for security
- **Health checks** for container monitoring
- **Dumb-init** for proper signal handling
- **Optimized JRE** for runtime (smaller than full JDK)

## Docker Compose

The `docker-compose.yml` file is already configured to use the frontend Dockerfile from the frontend folder. If you want to use the Dockerfile from this folder instead, update the frontend service configuration:

```yaml
frontend:
  build:
    context: .
    dockerfile: Dockerfile
  # ... rest of configuration
```

## Building and Running

1. **Using Docker Compose (Recommended):**
   ```bash
   docker-compose up --build frontend
   ```

2. **Using Docker directly:**
   ```bash
   # From the docker folder
   docker build -f Dockerfile -t cmlre-frontend ../frontend
   docker run -p 3000:80 cmlre-frontend
   ```

## Environment Variables

The frontend container supports these environment variables:
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:8080/api)
- `REACT_APP_ML_SERVICES_URL`: ML Services URL (default: http://localhost:8000)

## Ports

- **Container Port:** 80 (Nginx)
- **Host Port:** 3000 (mapped in docker-compose)

## Health Check

The container includes a health check that verifies the web server is responding on port 80.
