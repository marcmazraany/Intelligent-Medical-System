import { apiFetch } from "./client";
import { setTokens } from "../storage/tokens";

type SignupPayload = {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  password: string;
};

type SigninPayload = {
  email: string;
  password: string;
};

type AuthResponse = {
  accessToken: string;
  refreshToken?: string;
};

export async function signup(payload: SignupPayload): Promise<AuthResponse> {
  const data = (await apiFetch("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })) as AuthResponse;

  if (data?.accessToken) {
    await setTokens(data.accessToken, data.refreshToken);
  }

  return data;
}

export async function signin(payload: SigninPayload): Promise<AuthResponse> {
  const data = (await apiFetch("/api/auth/signin", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })) as AuthResponse;

  if (data?.accessToken) {
    await setTokens(data.accessToken, data.refreshToken);
  }

  return data;
}

export async function refresh(refreshToken: string): Promise<AuthResponse> {
  const data = (await apiFetch("/api/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refreshToken }),
  })) as AuthResponse;

  return data;
}
