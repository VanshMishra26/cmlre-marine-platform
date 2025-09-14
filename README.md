# CMLRE Marine Platform

An AI-enabled digital platform for marine data integration and analysis, designed for the Centre for Marine Living Resources and Ecology (CMLRE), Kochi.

## 🌊 Overview

The CMLRE Marine Platform is a comprehensive solution that integrates heterogeneous marine datasets from oceanography, taxonomy, morphology, and molecular biology into a unified system. It provides advanced analytics, machine learning capabilities, and interactive visualizations to support marine ecosystem assessment and sustainable resource management.

## 🏗️ Architecture

The platform follows a microservices architecture with the following components:

- **Backend**: Spring Boot REST API with MongoDB
- **ML Services**: Python FastAPI services for machine learning
- **Frontend**: React TypeScript application with Material-UI
- **Database**: MongoDB for data storage
- **Cache**: Redis for performance optimization
- **Monitoring**: Prometheus and Grafana for observability

## 🚀 Features

### Data Integration
- **Multi-source Data Ingestion**: Supports oceanographic, taxonomic, morphological, and molecular data
- **Automated Data Processing**: Standardized data formats and metadata tagging
- **Quality Control**: Built-in data validation and quality assessment
- **Real-time Processing**: Live data ingestion and analysis

### Machine Learning Capabilities
- **Otolith Analysis**: Advanced shape analysis and morphometrics using computer vision
- **Species Classification**: AI-powered species identification from images and molecular data
- **Molecular Analysis**: DNA sequence analysis, eDNA processing, and phylogenetic analysis
- **Cross-disciplinary Correlation**: Integrated analysis across different data types

### Visualization & Analytics
- **Interactive Dashboards**: Real-time data visualization and monitoring
- **Spatial Analysis**: Geographic distribution mapping and analysis
- **Time Series Analysis**: Trend analysis and forecasting
- **Statistical Analysis**: Comprehensive statistical tests and correlation analysis

### Data Management
- **Scalable Storage**: MongoDB-based data storage with optimized indexing
- **Data Export**: Multiple format support (CSV, Excel, JSON)
- **API Access**: RESTful APIs for data access and integration
- **Security**: JWT-based authentication and role-based access control

## 🛠️ Technology Stack

### Backend
- **Spring Boot 3.2**: Java 17 framework
- **MongoDB**: Document database
- **Redis**: Caching and session management
- **JWT**: Authentication and authorization
- **OpenAPI/Swagger**: API documentation

### ML Services
- **Python 3.11**: Core ML language
- **FastAPI**: High-performance web framework
- **OpenCV**: Computer vision processing
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow**: Deep learning models
- **BioPython**: Bioinformatics analysis

### Frontend
- **React 18**: Modern UI framework
- **TypeScript**: Type-safe development
- **Material-UI**: Component library
- **Recharts**: Data visualization
- **React Query**: Data fetching and caching

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy and load balancing
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

## 📦 Installation

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Java 17+ (for local development)
- Python 3.11+ (for local development)

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cmlre-marine-platform
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8080/api
   - ML Services: http://localhost:8000
   - API Documentation: http://localhost:8080/swagger-ui.html
   - Monitoring: http://localhost:3001 (Grafana)

### Local Development

1. **Backend Setup**
   ```bash
   cd backend
   ./mvnw spring-boot:run
   ```

2. **ML Services Setup**
   ```bash
   cd ml-services
   pip install -r requirements.txt
   python main.py
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

## 📊 Data Models

### Oceanographic Data
- Physical parameters (temperature, salinity, currents)
- Chemical parameters (dissolved oxygen, nutrients, pH)
- Biological parameters (primary production, species richness)
- Environmental conditions (weather, sea state)

### Taxonomic Data
- Species identification and classification
- Morphological characteristics
- Life history traits
- Ecological data and conservation status

### Morphological Data
- Otolith shape analysis and morphometrics
- Body measurements and ratios
- Geometric morphometric landmarks
- Traditional morphometric features

### Molecular Data
- DNA/RNA sequence analysis
- Environmental DNA (eDNA) processing
- Genetic diversity metrics
- Phylogenetic analysis

## 🔬 ML Services

### Otolith Analysis Service
- **Shape Feature Extraction**: Area, perimeter, circularity, aspect ratio
- **Geometric Morphometrics**: Landmark analysis, Procrustes analysis
- **Species Classification**: Machine learning-based species identification
- **Quality Assessment**: Image quality metrics and recommendations

### Taxonomy Classification Service
- **Morphological Classification**: Species identification from images
- **Molecular Classification**: DNA-based species identification
- **Integrated Classification**: Multi-source species identification
- **Taxonomic Hierarchy**: Full taxonomic classification

### Molecular Analysis Service
- **Sequence Analysis**: DNA/RNA sequence processing and analysis
- **eDNA Analysis**: Environmental DNA detection and quantification
- **Barcode Analysis**: DNA barcode identification and validation
- **Phylogenetic Analysis**: Evolutionary relationship analysis

### Data Integration Service
- **Correlation Analysis**: Cross-disciplinary parameter correlation
- **Statistical Testing**: Comprehensive statistical analysis
- **Clustering Analysis**: Data pattern identification
- **Dimensionality Reduction**: PCA, t-SNE, UMAP analysis

## 📈 API Documentation

### REST Endpoints

#### Oceanographic Data
- `GET /api/oceanography` - Get all oceanographic data
- `POST /api/oceanography` - Create new oceanographic data
- `GET /api/oceanography/{id}` - Get specific record
- `PUT /api/oceanography/{id}` - Update record
- `DELETE /api/oceanography/{id}` - Delete record
- `POST /api/oceanography/bulk-upload` - Bulk data upload
- `GET /api/oceanography/statistics` - Get statistical summary

#### Visualization
- `GET /api/visualization/oceanographic-trends` - Get oceanographic trends
- `GET /api/visualization/biodiversity-trends` - Get biodiversity trends
- `GET /api/visualization/spatial-distribution` - Get spatial distribution
- `GET /api/visualization/correlation-matrix` - Get correlation matrix

#### ML Services
- `POST /ml-services/otolith/analyze` - Analyze otolith shape
- `POST /ml-services/taxonomy/classify` - Classify species
- `POST /ml-services/molecular/analyze` - Analyze molecular data
- `POST /ml-services/integration/correlate` - Perform correlation analysis

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```env
MONGODB_URI=mongodb://localhost:27017/cmlre_marine_data
REDIS_URL=redis://localhost:6379
JWT_SECRET=your_jwt_secret_key
ML_SERVICES_URL=http://localhost:8000
```

#### ML Services
```env
MONGODB_URI=mongodb://localhost:27017/cmlre_marine_data
REDIS_HOST=localhost
REDIS_PORT=6379
MODEL_PATH=./models
```

#### Frontend
```env
REACT_APP_API_URL=http://localhost:8080/api
REACT_APP_ML_SERVICES_URL=http://localhost:8000
```

## 📊 Monitoring

### Health Checks
- Backend: `GET /api/health`
- ML Services: `GET /ml-services/health`

### Metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin123)

### Logs
- Application logs: `./logs/application.log`
- ML Services logs: Console output

## 🧪 Testing

### Backend Tests
```bash
cd backend
./mvnw test
```

### ML Services Tests
```bash
cd ml-services
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🚀 Deployment

### Production Deployment

1. **Configure environment variables**
2. **Build Docker images**
   ```bash
   docker-compose -f docker-compose.prod.yml build
   ```
3. **Deploy to production**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Scaling

- **Horizontal Scaling**: Add more container instances
- **Database Scaling**: MongoDB replica sets
- **Cache Scaling**: Redis cluster
- **Load Balancing**: Nginx configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔮 Future Enhancements

- **Advanced ML Models**: Deep learning for species identification
- **Real-time Streaming**: Apache Kafka integration
- **Mobile App**: React Native mobile application
- **Cloud Deployment**: AWS/Azure cloud deployment
- **API Gateway**: Kong or AWS API Gateway integration
- **Advanced Analytics**: Apache Spark for big data processing

## 📚 References

- [Spring Boot Documentation](https://spring.io/projects/spring-boot)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [Material-UI Documentation](https://mui.com/)

---

**CMLRE Marine Platform** - Empowering marine science through integrated data analytics and AI-driven insights.
