import http from 'k6/http';
import { check, sleep } from 'k6';

const backendUrl = (__ENV.BACKEND_URL || '').replace(/\/$/, '');

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '60s',
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<2000'],
  },
};

const payload = JSON.stringify({
  features: {
    longitude: -118.24,
    latitude: 34.05,
    housing_median_age: 30,
    total_rooms: 2400,
    total_bedrooms: 500,
    population: 1200,
    households: 400,
    median_income: 4.5,
    ocean_proximity: 'NEAR BAY',
  },
});

const params = {
  headers: {
    'Content-Type': 'application/json',
  },
};

export default function () {
  if (!backendUrl) {
    throw new Error('BACKEND_URL environment variable is required.');
  }

  const response = http.post(`${backendUrl}/api/predict`, payload, params);
  check(response, {
    'status is 200': (r) => r.status === 200,
    'prediction exists': (r) => Boolean(r.json('prediction')),
    'model_version exists': (r) => Boolean(r.json('model_version')),
    'request_id exists': (r) => Boolean(r.json('request_id')),
  });
  sleep(1);
}
