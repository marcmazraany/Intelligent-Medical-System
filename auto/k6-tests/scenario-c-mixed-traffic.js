import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

const errorRate = new Rate('errors');
const searchDuration = new Trend('search_duration', true);
const cacheHits = new Counter('cache_hits');
const cacheMisses = new Counter('cache_misses');

export const options = {
    stages: [
        { duration: '10s', target: 20 },
        { duration: '2m', target: 50 },
        { duration: '10s', target: 0 },
    ],
    thresholds: {
        'http_req_duration': ['p(95)<1000'],
        'errors': ['rate<0.05'],
    },
};

// 80% cache hit (common medications), 20% cache miss (random)
const commonMeds = ['Panadol', 'Augmentin', 'Ventolin', 'Aspirin'];
const rareMeds = ['RareDrug1', 'RareDrug2', 'RareDrug3', 'RareDrug4'];

export function setup() {
    // Warm cache with common medications
    console.log('Warming cache with common medications...');
    commonMeds.forEach(med => {
        http.get(`http://localhost:8080/api/medications/search?name=${med}`);
        sleep(0.5);
    });
}

export default function () {
    const isCacheHit = Math.random() < 0.8; // 80% cache hit
    const medication = isCacheHit
        ? commonMeds[Math.floor(Math.random() * commonMeds.length)]
        : rareMeds[Math.floor(Math.random() * rareMeds.length)];

    if (isCacheHit) {
        cacheHits.add(1);
    } else {
        cacheMisses.add(1);
    }

    const start = new Date();
    const res = http.get(`http://localhost:8080/api/medications/search?name=${medication}`);
    const duration = new Date() - start;

    searchDuration.add(duration);

    const success = check(res, {
        'status is 200': (r) => r.status === 200,
    });

    errorRate.add(!success);

    sleep(0.1);
}

export function handleSummary(data) {
    return {
        'results/scenario-c-mixed-traffic.json': JSON.stringify(data, null, 2),
    };
}