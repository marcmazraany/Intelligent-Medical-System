# IntelliMeds Backend (Clean)

Spring Boot backend designed to match the current React/Capacitor frontend types.

## Features (v1)
- JWT auth (access + refresh)
- Profile: GET/PUT `/api/profile`
- Medications CRUD: `/api/medications`
- Alerts CRUD: `/api/alerts`
- Flyway migrations (PostgreSQL)

## Run locally
1) Create a Postgres DB (or use Supabase) and set env vars:

```bash
export DB_URL='jdbc:postgresql://localhost:5432/intellimeds'
export DB_USER='postgres'
export DB_PASS='postgres'
export JWT_SECRET='change-me-super-secret-change-me-super-secret'
export CORS_ALLOWED_ORIGINS='http://localhost:5173,http://localhost:8100'
```

2) Start:
```bash
./gradlew bootRun
```

## Auth
- `POST /api/auth/signup` -> `{accessToken, refreshToken, profile}`
- `POST /api/auth/signin` -> `{accessToken, refreshToken, profile}`
- `POST /api/auth/refresh` -> `{accessToken}`

Send `Authorization: Bearer <accessToken>` on protected endpoints.
