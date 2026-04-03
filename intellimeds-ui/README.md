# IntelliMeds (UI-only)

This is a **frontend-only** (no backend) refactor of your IntelliMeds UI prototype.

## What changed vs your original single-file prototype
- Split into **pages** (`src/pages/*`) and reusable **components** (`src/components/*`).
- Switched to **React Router** (HashRouter) for clean page navigation (Capacitor-friendly).
- Removed **all Gemini / AI API calls** (AI tab is UI-only with mock responses).
- Added simple **localStorage persistence** for profile + medication list.

## Run locally
```bash
npm install
npm run dev
```

## Routes
- `#/` Home
- `#/inventory` Inventory
- `#/pharmacies` Find Pharmacies
- `#/alerts` Alerts
- `#/ai` AI Help (UI-only)

## Notes for Capacitor later
Hash-based routing (`HashRouter`) avoids Android Studio / WebView deep-link issues. When you move to Capacitor, you can keep this as-is, or switch to BrowserRouter with proper URL handling.


## Android (USB) backend connection (8080)

If you test on a **real Android phone** via USB, run:

```bash
adb reverse tcp:8080 tcp:8080
```

Then the app will call the backend using `http://localhost:8080` (configured in `.env`).
