import { apiFetch } from "./client";

export type MedicationRequest = {
  name: string;
  dosage: string;
  expiryDate: string;
  frequency: string;
  quantity?: number;
  reminderTimes: string[];
  status?: string;
  notes?: string;
};

export type MedicationDto = {
  id: string;
  name: string;
  dosage: string;
  expiryDate: string;
  frequency: string;
  quantity?: number;
  reminderTimes: string[];
  status?: string;
  notes?: string;
};

export type MedicationScanItem = {
  medication?: MedicationDto;
  action?: string;
  scan?: {
    source?: string;
    gtin?: string;
    name?: string;
    manufacturer?: string;
    dosage?: string | null;
    quantity?: string | null;
    form?: string | null;
    expiryDate?: string | null;
  };
};

export type MedicationScanBatchResponse = {
  source?: string;
  detectedCount?: number;
  items?: MedicationScanItem[];
};

export async function listMedications(): Promise<unknown> {
  return apiFetch("/api/medications", { method: "GET", auth: true });
}

export async function createMedication(
  req: MedicationRequest
): Promise<unknown> {
  return apiFetch("/api/medications", {
    method: "POST",
    auth: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function updateMedication(
  id: string,
  req: MedicationRequest
): Promise<unknown> {
  return apiFetch(`/api/medications/${id}`, {
    method: "PUT",
    auth: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function deleteMedication(id: string): Promise<unknown> {
  return apiFetch(`/api/medications/${id}`, {
    method: "DELETE",
    auth: true,
  });
}

export async function scanMedication(file: File): Promise<unknown> {
  const formData = new FormData();
  formData.append("file", file);

  return apiFetch("/api/medications/scan", {
    method: "POST",
    auth: true,
    body: formData,
  });
}