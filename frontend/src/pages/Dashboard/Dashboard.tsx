import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
  LinearProgress,
  Chip,
  Alert,
  Button,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Water as WaterIcon,
  BugReport as BugReportIcon,
  Science as ScienceIcon,
  Biotech as BiotechIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useQuery } from 'react-query';
import { apiService } from '../../services/apiService';

interface DashboardStats {
  totalRecords: number;
  oceanographicRecords: number;
  taxonomicRecords: number;
  morphologicalRecords: number;
  molecularRecords: number;
  dataQuality: number;
  lastUpdated: string;
}

interface TimeSeriesData {
  date: string;
  temperature: number;
  salinity: number;
  species: number;
}

interface SpeciesDistribution {
  name: string;
  value: number;
  color: string;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [speciesDistribution, setSpeciesDistribution] = useState<SpeciesDistribution[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch dashboard data
  const { data: dashboardData, isLoading, error: queryError, refetch } = useQuery(
    'dashboardData',
    async () => {
      const [statsResponse, timeSeriesResponse, speciesResponse] = await Promise.all([
        apiService.get('/oceanography/statistics'),
        apiService.get('/visualization/oceanographic-trends?parameter=surface_temperature&period=monthly'),
        apiService.get('/visualization/species-distribution')
      ]);
      
      return {
        stats: statsResponse.data,
        timeSeries: timeSeriesResponse.data,
        species: speciesResponse.data
      };
    },
    {
      refetchInterval: 30000, // Refetch every 30 seconds
      onError: (error) => {
        setError('Failed to load dashboard data');
        console.error('Dashboard data error:', error);
      }
    }
  );

  useEffect(() => {
    if (dashboardData) {
      setStats(dashboardData.stats);
      setTimeSeriesData(dashboardData.timeSeries?.timeSeries || []);
      setSpeciesDistribution(dashboardData.species?.distribution || []);
      setLoading(false);
    }
  }, [dashboardData]);

  const handleRefresh = () => {
    refetch();
  };

  const StatCard: React.FC<{
    title: string;
    value: number | string;
    icon: React.ReactNode;
    color: string;
    trend?: number;
  }> = ({ title, value, icon, color, trend }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="h6">
              {title}
            </Typography>
            <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
              {value}
            </Typography>
            {trend !== undefined && (
              <Box display="flex" alignItems="center" mt={1}>
                <TrendingUpIcon 
                  sx={{ 
                    color: trend > 0 ? 'success.main' : 'error.main',
                    mr: 0.5,
                    fontSize: 16
                  }} 
                />
                <Typography 
                  variant="body2" 
                  color={trend > 0 ? 'success.main' : 'error.main'}
                >
                  {trend > 0 ? '+' : ''}{trend}%
                </Typography>
              </Box>
            )}
          </Box>
          <Box
            sx={{
              backgroundColor: color,
              borderRadius: '50%',
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>
          Dashboard
        </Typography>
        <Alert severity="error" action={
          <Button color="inherit" size="small" onClick={handleRefresh}>
            Retry
          </Button>
        }>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          Marine Data Dashboard
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={isLoading}
        >
          Refresh
        </Button>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Records"
            value={stats?.totalRecords || 0}
            icon={<WaterIcon sx={{ color: 'white' }} />}
            color="#1976d2"
            trend={5.2}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Oceanographic"
            value={stats?.oceanographicRecords || 0}
            icon={<WaterIcon sx={{ color: 'white' }} />}
            color="#00c853"
            trend={3.1}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Taxonomic"
            value={stats?.taxonomicRecords || 0}
            icon={<BugReportIcon sx={{ color: 'white' }} />}
            color="#ff9800"
            trend={7.8}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Data Quality"
            value={`${stats?.dataQuality || 0}%`}
            icon={<ScienceIcon sx={{ color: 'white' }} />}
            color="#9c27b0"
            trend={2.3}
          />
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} mb={3}>
        {/* Time Series Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Oceanographic Trends
              </Typography>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="temperature" 
                      stroke="#1976d2" 
                      strokeWidth={2}
                      name="Temperature (°C)"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="salinity" 
                      stroke="#00c853" 
                      strokeWidth={2}
                      name="Salinity (PSU)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Species Distribution Pie Chart */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Species Distribution
              </Typography>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={speciesDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {speciesDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Data Quality and Recent Activity */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Data Quality Metrics
              </Typography>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Overall Quality</Typography>
                  <Typography variant="body2">{stats?.dataQuality || 0}%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={stats?.dataQuality || 0} 
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Approved Records</Typography>
                  <Typography variant="body2">85%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={85} 
                  color="success"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">Pending Review</Typography>
                  <Typography variant="body2">12%</Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={12} 
                  color="warning"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              <Box>
                <Box display="flex" alignItems="center" mb={2}>
                  <Chip 
                    label="New" 
                    size="small" 
                    color="primary" 
                    sx={{ mr: 1 }}
                  />
                  <Typography variant="body2">
                    150 oceanographic records uploaded
                  </Typography>
                </Box>
                <Box display="flex" alignItems="center" mb={2}>
                  <Chip 
                    label="Updated" 
                    size="small" 
                    color="success" 
                    sx={{ mr: 1 }}
                  />
                  <Typography variant="body2">
                    25 taxonomic records verified
                  </Typography>
                </Box>
                <Box display="flex" alignItems="center" mb={2}>
                  <Chip 
                    label="Analysis" 
                    size="small" 
                    color="info" 
                    sx={{ mr: 1 }}
                  />
                  <Typography variant="body2">
                    Otolith analysis completed for 10 specimens
                  </Typography>
                </Box>
                <Box display="flex" alignItems="center">
                  <Chip 
                    label="Alert" 
                    size="small" 
                    color="warning" 
                    sx={{ mr: 1 }}
                  />
                  <Typography variant="body2">
                    3 data quality issues detected
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
