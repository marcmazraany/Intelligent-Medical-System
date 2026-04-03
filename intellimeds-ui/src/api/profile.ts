import { apiFetch } from "./client";

export type ProfileUpdateDto = {
  firstName: string;
  lastName: string;

  dob?: string; // yyyy-mm-dd
  gender?: string;
  height?: string;
  weight?: string;
  allergies?: string;
  bloodType?: string;
  chronicConditions?: string;
  notes?: string;
};

export async function getProfile() {
  return apiFetch("/api/profile", { auth: true });
}

export async function updateProfile(dto: ProfileUpdateDto) {
  return apiFetch("/api/profile", {
    method: "PUT",
    auth: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(dto),
  });
}
