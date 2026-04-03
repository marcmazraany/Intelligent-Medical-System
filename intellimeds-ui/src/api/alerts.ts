import { apiFetch } from "./client";

export type AlertRequest = {
  medicationName: string;
  maxPrice: number;
  emailEnabled: boolean;
  maxDistance?: number; // km, optional
};

export async function listAlerts(): Promise<unknown> {
  return apiFetch("/api/alerts", { method: "GET", auth: true });
}

export async function createAlert(req: AlertRequest): Promise<unknown> {
  return apiFetch("/api/alerts", {
    method: "POST",
    auth: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function patchAlert(
  id: string,
  req: Partial<AlertRequest> & { active?: boolean; status?: string }
): Promise<unknown> {
  return apiFetch(`/api/alerts/${id}`, {
    method: "PATCH",
    auth: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function deleteAlert(id: string): Promise<unknown> {
  return apiFetch(`/api/alerts/${id}`, { method: "DELETE", auth: true });
}
