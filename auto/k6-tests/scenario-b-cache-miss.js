import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const searchDuration = new Trend('search_duration', true);

export const options = {
    stages: [
        { duration: '10s', target: 5 },   // Warm up slowly
        { duration: '1m', target: 20 },   // Moderate load (cache miss is heavy)
        { duration: '10s', target: 0 },   // Cool down
    ],
    thresholds: {
        'http_req_duration': ['p(95)<3000'], // 95% under 3 seconds
        'errors': ['rate<0.1'],               // Error rate under 10%
    },
};

// Generate random medication names to force cache misses
const medications = [
    'TestMed1', 'TestMed2', 'TestMed3', 'TestMed4', 'TestMed5',
    'RandomDrug1', 'RandomDrug2', 'RandomDrug3', 'RandomDrug4', 'RandomDrug5'
];

export default function () {
    const medication = medications[Math.floor(Math.random() * medications.length)];
    const start = new Date();

    // This will trigger real-time pharmacy ping (cache miss)
    const res = http.get(`http://localhost:8080/api/medications/search?name=${medication}`);
    const duration = new Date() - start;

    searchDuration.add(duration);

    const success = check(res, {
        'status is 200': (r) => r.status === 200,
        'response time acceptable': (r) => r.timings.duration < 5000,
    });

    errorRate.add(!success);

    sleep(0.5); // Longer delay for cache-miss scenario
}

export function handleSummary(data) {
    return {
        'results/scenario-b-cache-miss.json': JSON.stringify(data, null, 2),
    };
}