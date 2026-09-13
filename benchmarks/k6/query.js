import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '30s',
  thresholds: {
    http_req_failed: ['rate<0.02'],
    http_req_duration: ['p(95)<500'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const TOKEN = __ENV.ACCESS_TOKEN || '';
const WORKSPACE_ID = __ENV.WORKSPACE_ID || '';

export default function () {
  const payload = JSON.stringify({ workspace_id: WORKSPACE_ID, query: 'What is the PTO policy?' });
  const params = {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `******
    },
  };
  const res = http.post(`${BASE_URL}/v1/query`, payload, params);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(1);
}
