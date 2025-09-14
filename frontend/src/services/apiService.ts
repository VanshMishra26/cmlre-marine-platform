import axios, { AxiosInstance, AxiosResponse } from 'axios';

class ApiService {
  private api: AxiosInstance;
  private mlApi: AxiosInstance;

  constructor() {
    // Main API (Spring Boot backend)
    this.api = axios.create({
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8080/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // ML Services API (Python FastAPI)
    this.mlApi = axios.create({
      baseURL: process.env.REACT_APP_ML_SERVICES_URL || 'http://localhost:8000',
      timeout: 60000, // Longer timeout for ML operations
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for authentication
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Add response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );

    this.mlApi.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('ML API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  // Generic HTTP methods
  async get<T = any>(url: string, params?: any): Promise<AxiosResponse<T>> {
    return this.api.get(url, { params });
  }

  async post<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.post(url, data);
  }

  async put<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.put(url, data);
  }

  async delete<T = any>(url: string): Promise<AxiosResponse<T>> {
    return this.api.delete(url);
  }

  // ML Services methods
  async mlGet<T = any>(url: string, params?: any): Promise<AxiosResponse<T>> {
    return this.mlApi.get(url, { params });
  }

  async mlPost<T = any>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.mlApi.post(url, data);
  }

  // File upload methods
  async uploadFile(url: string, file: File, onProgress?: (progress: number) => void): Promise<AxiosResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.api.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
  }

  // Oceanographic Data API
  oceanography = {
    getAll: (params?: any) => this.get('/oceanography', params),
    getById: (id: string) => this.get(`/oceanography/${id}`),
    create: (data: any) => this.post('/oceanography', data),
    update: (id: string, data: any) => this.put(`/oceanography/${id}`, data),
    delete: (id: string) => this.delete(`/oceanography/${id}`),
    bulkUpload: (file: File, dataSource?: string) => {
      const formData = new FormData();
      formData.append('file', file);
      if (dataSource) formData.append('dataSource', dataSource);
      return this.api.post('/oceanography/bulk-upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    },
    getStatistics: (params?: any) => this.get('/oceanography/statistics', params),
    getByLocation: (params: any) => this.get('/oceanography/location', params),
    getQualityMetrics: () => this.get('/oceanography/quality-metrics'),
    export: (params: any) => this.get('/oceanography/export', { params, responseType: 'blob' }),
    getTimeSeries: (params: any) => this.get('/oceanography/time-series', params),
  };

  // Taxonomy Data API
  taxonomy = {
    getAll: (params?: any) => this.get('/taxonomy', params),
    getById: (id: string) => this.get(`/taxonomy/${id}`),
    create: (data: any) => this.post('/taxonomy', data),
    update: (id: string, data: any) => this.put(`/taxonomy/${id}`, data),
    delete: (id: string) => this.delete(`/taxonomy/${id}`),
    search: (query: string) => this.get('/taxonomy/search', { params: { q: query } }),
    getBySpecies: (speciesName: string) => this.get(`/taxonomy/species/${speciesName}`),
    getStatistics: () => this.get('/taxonomy/statistics'),
  };

  // Morphology Data API
  morphology = {
    getAll: (params?: any) => this.get('/morphology', params),
    getById: (id: string) => this.get(`/morphology/${id}`),
    create: (data: any) => this.post('/morphology', data),
    update: (id: string, data: any) => this.put(`/morphology/${id}`, data),
    delete: (id: string) => this.delete(`/morphology/${id}`),
    analyzeOtolith: (file: File) => {
      const formData = new FormData();
      formData.append('image', file);
      return this.mlPost('/otolith/analyze', formData);
    },
    classifySpecies: (data: any) => this.mlPost('/otolith/classify', data),
  };

  // Molecular Data API
  molecular = {
    getAll: (params?: any) => this.get('/molecular', params),
    getById: (id: string) => this.get(`/molecular/${id}`),
    create: (data: any) => this.post('/molecular', data),
    update: (id: string, data: any) => this.put(`/molecular/${id}`, data),
    delete: (id: string) => this.delete(`/molecular/${id}`),
    analyzeSequence: (data: any) => this.mlPost('/molecular/analyze', data),
    analyzeEdna: (data: any) => this.mlPost('/molecular/edna', data),
    getStatistics: () => this.get('/molecular/statistics'),
  };

  // Visualization API
  visualization = {
    getOceanographicTrends: (params: any) => this.get('/visualization/oceanographic-trends', params),
    getBiodiversityTrends: (params: any) => this.get('/visualization/biodiversity-trends', params),
    getSpatialDistribution: (params: any) => this.get('/visualization/spatial-distribution', params),
    getCorrelationMatrix: (params: any) => this.get('/visualization/correlation-matrix', params),
    getDepthProfile: (params: any) => this.get('/visualization/depth-profile', params),
    getSpeciesDistribution: (params: any) => this.get('/visualization/species-distribution', params),
    getEnvironmentalFactors: (params: any) => this.get('/visualization/environmental-factors', params),
    getDataQualityDashboard: () => this.get('/visualization/data-quality-dashboard'),
    getRealTimeMonitoring: (params: any) => this.get('/visualization/real-time-monitoring', params),
    createCustomVisualization: (config: any) => this.post('/visualization/custom', config),
  };

  // Analysis API
  analysis = {
    correlateData: (data: any) => this.mlPost('/integration/correlate', data),
    createVisualization: (data: any) => this.mlPost('/integration/visualize', data),
    getInsights: (params: any) => this.get('/analysis/insights', params),
    getRecommendations: (params: any) => this.get('/analysis/recommendations', params),
  };

  // Authentication API
  auth = {
    login: (credentials: { username: string; password: string }) => 
      this.post('/auth/login', credentials),
    logout: () => this.post('/auth/logout'),
    refreshToken: () => this.post('/auth/refresh'),
    getProfile: () => this.get('/auth/profile'),
  };

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      await this.get('/health');
      return true;
    } catch {
      return false;
    }
  }

  async mlHealthCheck(): Promise<boolean> {
    try {
      await this.mlGet('/health');
      return true;
    } catch {
      return false;
    }
  }
}

export const apiService = new ApiService();
