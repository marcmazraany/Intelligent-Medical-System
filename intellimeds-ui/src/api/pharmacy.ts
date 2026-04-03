import { apiFetch } from "./client";

// ── Types that mirror what IntelliMeds backend returns ──────────────

export type PharmacySearchResult = {
  pharmacyName: string;
  address: string | null;
  latitude: number;
  longitude: number;
  distanceKm: number;
  travelTimeMinutes: number | null;
  stockQuantity: number;
  price: number;
  currency: string;
  inStock: boolean;
  googleMapsUrl: string | null;
  lastUpdated: string;
  pharmacyPhone: string | null;
};

export type MedicationAvailabilityResponse = {
  medicationName: string;
  dosage: string;
  totalPharmaciesChecked: number;
  totalPharmaciesWithStock: number;
  pharmacies: PharmacySearchResult[];
};

// ── Search (cache-first) ─────────────────────────────────────────────

export async function searchPharmacies(
  name: string,
  latitude?: number,
  longitude?: number,
  maxResults = 10
): Promise<MedicationAvailabilityResponse> {
  const params = new URLSearchParams({ name, maxResults: String(maxResults) });
  if (latitude  != null) params.set("latitude",  String(latitude));
  if (longitude != null) params.set("longitude", String(longitude));

  return apiFetch(`/api/pharmacy-finder/search?${params.toString()}`, {
    method: "GET",
    auth: true,
  }) as Promise<MedicationAvailabilityResponse>;
}

// ── Force live ping (bypass cache) ──────────────────────────────────

export async function pingPharmacies(
  name: string,
  latitude?: number,
  longitude?: number,
  maxDistanceKm = 50
): Promise<MedicationAvailabilityResponse> {
  const params = new URLSearchParams({
    name,
    maxDistanceKm: String(maxDistanceKm),
  });
  if (latitude  != null) params.set("latitude",  String(latitude));
  if (longitude != null) params.set("longitude", String(longitude));

  return apiFetch(`/api/pharmacy-finder/ping?${params.toString()}`, {
    method: "POST",
    auth: true,
  }) as Promise<MedicationAvailabilityResponse>;
}
