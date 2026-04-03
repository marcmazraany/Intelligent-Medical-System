import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const searchDuration = new Trend('search_duration', true);

export const options = {
    stages: [
        { duration: '10s', target: 10 },  // Warm up
        { duration: '1m', target: 50 },   // Load test
        { duration: '10s', target: 0 },   // Cool down
    ],
    thresholds: {
        'http_req_duration': ['p(95)<500'], // 95% of requests under 500ms
        'errors': ['rate<0.05'],             // Error rate under 5%
    },
};

export function setup() {
    console.log("Warming cache strongly...");
    for (let i = 0; i < 30; i++) {
        http.get("http://localhost:8080/api/medications/search?name=Panadol");
        sleep(0.1);
    }
}


export default function () {
    const start = new Date();
    const res = http.get('http://localhost:8080/api/medications/search?name=Panadol');
    const duration = new Date() - start;

    searchDuration.add(duration);

    const success = check(res, {
        'status is 200': (r) => r.status === 200,
        'has pharmacies': (r) => {
            const body = JSON.parse(r.body);
            return body.pharmacies && body.pharmacies.length > 0;
        },
        'response time OK': (r) => r.timings.duration < 1000,
    });

    errorRate.add(!success);

    sleep(0.1); // Small delay between requests
}

export function handleSummary(data) {
    return {
        'results/scenario-a-cache-hit.json': JSON.stringify(data, null, 2),
    };
}