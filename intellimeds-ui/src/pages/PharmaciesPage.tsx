import React, { useState } from "react";
import {
  Bell,
  ExternalLink,
  MapPin,
  Navigation,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { Header } from "@/components/Header";
import { useNavigate } from "react-router-dom";
import type { PharmacySearchResult } from "@/api/pharmacy";
import { pingPharmacies, searchPharmacies } from "@/api/pharmacy";
import { createAlert } from "@/api/alerts";

function formatPrice(price: number, currency: string): string {
  if (!currency || currency === "LBP") {
    return `${Math.round(price).toLocaleString()} LBP`;
  }
  return `${price.toFixed(2)} ${currency}`;
}

// ── Single pharmacy card ─────────────────────────────────────────────
function PharmacyCard({
  pharmacy,
  onCreateAlert,
}: {
  pharmacy: PharmacySearchResult;
  onCreateAlert: (p: PharmacySearchResult) => void;
}) {
  return (
    <div className="bg-slate-50 rounded-3xl p-5 border border-slate-200">
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <p className="text-slate-800 font-bold text-lg truncate">
            {pharmacy.pharmacyName}
          </p>
          {pharmacy.address && (
            <p className="text-slate-500 text-sm mt-1 flex items-center gap-2">
              <MapPin className="w-4 h-4 shrink-0" />
              {pharmacy.address}
            </p>
          )}
          <p className="text-slate-400 text-xs mt-2">
            Updated {pharmacy.lastUpdated}
          </p>
        </div>
        <div className="text-right shrink-0 ml-3">
          <p className="text-slate-800 font-bold">
            {formatPrice(pharmacy.price, pharmacy.currency)}
          </p>
          <p className="text-slate-500 text-sm">
            {pharmacy.distanceKm.toFixed(1)} km
          </p>
          {pharmacy.travelTimeMinutes != null && (
            <p className="text-slate-400 text-xs">
              ~{pharmacy.travelTimeMinutes} min
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mt-4">
        <div className="bg-white rounded-2xl p-3">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
            Stock
          </p>
          <p className="text-sm font-semibold text-slate-700 mt-1">
            {pharmacy.stockQuantity} units
          </p>
        </div>
        <div className="bg-white rounded-2xl p-3">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
            Availability
          </p>
          <p className={`text-sm font-semibold mt-1 ${pharmacy.inStock ? "text-emerald-600" : "text-red-600"}`}>
            {pharmacy.inStock ? "In stock" : "Out of stock"}
          </p>
        </div>
      </div>

      <div className="mt-4 flex gap-2">
        {/* Directions — always shown */}
        {pharmacy.googleMapsUrl && (
          <a
            href={pharmacy.googleMapsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 bg-white hover:bg-slate-100 text-slate-700 font-semibold py-3 px-4 rounded-2xl flex items-center justify-center gap-2 transition-colors"
          >
            Directions <ExternalLink className="w-4 h-4" />
          </a>
        )}

        {/* Alert me — ONLY shown when OUT OF STOCK */}
        {!pharmacy.inStock && (
          <button
            onClick={() => onCreateAlert(pharmacy)}
            className="bg-orange-50 hover:bg-orange-100 text-orange-600 font-semibold py-3 px-4 rounded-2xl flex items-center gap-2 transition-colors"
          >
            <Bell className="w-4 h-4" />
            Alert me
          </button>
        )}
      </div>

      {pharmacy.pharmacyPhone && (
        <p className="mt-3 text-slate-400 text-xs text-center">
          📞 {pharmacy.pharmacyPhone}
        </p>
      )}
    </div>
  );
}

// ── Create alert modal — floated well above the bottom nav ───────────
function CreateAlertModal({
  medicationName,
  onClose,
  onCreated,
}: {
  medicationName: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [maxPrice, setMaxPrice]         = useState("");
  const [maxDistance, setMaxDistance]   = useState("10");
  const [emailEnabled, setEmailEnabled] = useState(true);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState<string | null>(null);

  const handleCreate = async () => {
    const price = parseFloat(maxPrice);
    if (isNaN(price) || price <= 0) {
      setError("Enter a valid max price.");
      return;
    }
    try {
      setLoading(true);
      setError(null);
      await createAlert({
        medicationName,
        maxPrice: price,
        emailEnabled,
        maxDistance: parseFloat(maxDistance) || undefined,
      });
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create alert.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 px-4">
      {/* mb-24 = 96px gap above the bottom nav bar (~80px tall) */}
      <div className="w-full max-w-sm mb-24">
        <div className="bg-white rounded-3xl p-6 shadow-2xl">
          <div className="flex items-center justify-between mb-3">
            <p className="font-bold text-slate-800 text-lg">Notify me when in stock</p>
            <button onClick={onClose} className="p-1 rounded-full hover:bg-slate-100">
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>

          <p className="text-slate-500 text-sm mb-5">
            You'll get a notification when{" "}
            <span className="font-semibold text-slate-700">{medicationName}</span>{" "}
            is back in stock nearby.
          </p>

          <div className="space-y-3">
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                Max price (LBP)
              </label>
              <input
                type="number"
                value={maxPrice}
                onChange={(e) => setMaxPrice(e.target.value)}
                placeholder="e.g. 50000"
                className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-2xl px-4 py-3 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>

            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                Max distance (km)
              </label>
              <input
                type="number"
                value={maxDistance}
                onChange={(e) => setMaxDistance(e.target.value)}
                placeholder="e.g. 10"
                className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-2xl px-4 py-3 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>

            <div className="flex items-center justify-between bg-slate-50 rounded-2xl px-4 py-3">
              <p className="text-sm font-semibold text-slate-700">Email notification</p>
              <button
                type="button"
                onClick={() => setEmailEnabled((v) => !v)}
                className={`w-12 h-6 rounded-full transition-colors relative ${emailEnabled ? "bg-blue-600" : "bg-slate-300"}`}
              >
                <span className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${emailEnabled ? "left-7" : "left-1"}`} />
              </button>
            </div>
          </div>

          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

          <button
            onClick={() => void handleCreate()}
            disabled={loading}
            className="mt-5 w-full bg-blue-600 text-white font-bold py-4 rounded-2xl shadow-lg shadow-blue-500/30 disabled:opacity-60"
          >
            {loading ? "Creating..." : "Notify me"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────
export function PharmaciesPage() {
  const navigate = useNavigate();

  const [query, setQuery]               = useState("");
  const [results, setResults]           = useState<PharmacySearchResult[]>([]);
  const [searching, setSearching]       = useState(false);
  const [searched, setSearched]         = useState(false);
  const [error, setError]               = useState<string | null>(null);
  const [totalChecked, setTotalChecked] = useState(0);
  const [alertTarget, setAlertTarget]   = useState<string | null>(null);
  const [alertCreated, setAlertCreated] = useState(false);

  const getUserLocation = (): Promise<{ lat: number; lon: number } | null> =>
    new Promise((resolve) => {
      if (!navigator.geolocation) return resolve(null);
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
        () => resolve(null),
        { timeout: 5000 }
      );
    });

  const doSearch = async (forcePing = false) => {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    setSearched(false);

    try {
      const loc = await getUserLocation();
      const fn = forcePing ? pingPharmacies : searchPharmacies;
      const data = await fn(query.trim(), loc?.lat, loc?.lon);
      setResults(data.pharmacies ?? []);
      setTotalChecked(data.totalPharmaciesChecked);
      setSearched(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed. Please try again.");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="pb-24">
      <Header
        title="IntelliMeds"
        subtitle="Find medications at nearby pharmacies instantly"
        onProfileClick={() => navigate("/profile")}
      />

      <div className="px-5 mt-6">
        <div className="bg-white rounded-3xl p-6 shadow-xl border border-slate-100 space-y-4">
          <div>
            <p className="text-slate-800 font-bold text-lg">Search medication availability</p>
            <p className="text-slate-500 text-sm mt-1">
              Type a medication name and we'll check nearby pharmacies.
            </p>
          </div>

          <div className="relative">
            <Search className="w-5 h-5 text-slate-400 absolute left-4 top-1/2 -translate-y-1/2" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void doSearch()}
              placeholder="e.g., Panadol Extra"
              className="w-full bg-slate-50 border border-slate-200 rounded-2xl pl-12 pr-4 py-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => void doSearch(false)}
              disabled={searching || !query.trim()}
              className="flex-1 bg-blue-600 text-white font-bold py-4 rounded-2xl shadow-lg shadow-blue-500/30 active:scale-[0.99] disabled:opacity-60"
            >
              {searching ? "Searching..." : "Search"}
            </button>

            <button
              onClick={() => void doSearch(true)}
              disabled={searching || !query.trim()}
              title="Force live refresh"
              className="bg-slate-100 hover:bg-slate-200 text-slate-600 font-bold py-4 px-4 rounded-2xl active:scale-[0.99] disabled:opacity-60 transition-colors"
            >
              <RefreshCw className={`w-5 h-5 ${searching ? "animate-spin" : ""}`} />
            </button>
          </div>

          <p className="text-xs text-slate-400 flex items-center gap-1">
            <Navigation className="w-3 h-3" />
            Results are sorted by distance from your location.
          </p>
        </div>

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-2xl px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {searched && (
          <div className="mt-6 space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-slate-800 font-bold">
                {results.length > 0 ? `${results.length} pharmacies found` : "No pharmacies found"}
              </p>
              {totalChecked > 0 && (
                <p className="text-slate-400 text-xs">{totalChecked} pharmacies checked</p>
              )}
            </div>

            {results.length === 0 ? (
              <div className="bg-white rounded-3xl p-8 text-center border border-slate-100 shadow-sm">
                <Search className="w-8 h-8 text-slate-300 mx-auto" />
                <p className="text-slate-800 font-bold mt-3">Not found nearby</p>
                <p className="text-slate-500 text-sm mt-1">
                  Try the live refresh button or check the spelling.
                </p>
              </div>
            ) : (
              results.map((p, i) => (
                <PharmacyCard
                  key={`${p.pharmacyName}-${i}`}
                  pharmacy={p}
                  onCreateAlert={() => setAlertTarget(query.trim())}
                />
              ))
            )}
          </div>
        )}
      </div>

      {alertTarget && (
        <CreateAlertModal
          medicationName={alertTarget}
          onClose={() => setAlertTarget(null)}
          onCreated={() => {
            setAlertTarget(null);
            setAlertCreated(true);
            setTimeout(() => setAlertCreated(false), 3000);
          }}
        />
      )}

      {alertCreated && (
        <div className="fixed bottom-28 left-1/2 -translate-x-1/2 bg-emerald-600 text-white text-sm font-semibold px-6 py-3 rounded-2xl shadow-lg z-50 whitespace-nowrap">
          ✓ We'll notify you when it's back in stock
        </div>
      )}
    </div>
  );
}
