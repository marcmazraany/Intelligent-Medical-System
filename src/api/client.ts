/// <reference types="vite/client" />

import { getTokens, setTokens, clearTokens } from "../storage/tokens";
import { refresh } from "./auth";

type ApiFetchOptions = RequestInit & {
  auth?: boolean;
};

const defaultUrl = import.meta.env.DEV ? "http://localhost:8080" : "";
const baseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) || defaultUrl;

if (!baseUrl) {
  throw new Error(
    "API base URL is not configured. Set VITE_API_BASE_URL in your env."
  );
}

const refreshPath = "/api/auth/refresh";
const normalizedBaseUrl = baseUrl.replace(/\/+$/, "");
let refreshPromise: Promise<string | null> | null = null;

function getErrorMessage(data: unknown, response: Response): string {
  return (
    (typeof data === "object" &&
      data !== null &&
      "message" in data &&
      typeof (data as { message?: unknown }).message === "string" &&
      (data as { message: string }).message) ||
    (typeof data === "object" &&
      data !== null &&
      "error" in data &&
      typeof (data as { error?: unknown }).error === "string" &&
      (data as { error: string }).error) ||
    response.statusText ||
    `Request failed with status ${response.status}`
  );
}

async function refreshAccessToken(): Promise<string | null> {
  if (!refreshPromise) {
    refreshPromise = (async () => {
      const { refreshToken } = await getTokens();
      if (!refreshToken) return null;

      try {
        const data = await refresh(refreshToken);
        const accessToken =
          data && typeof data === "object" && "accessToken" in data
            ? (data as { accessToken?: string }).accessToken
            : undefined;
        const nextRefreshToken =
          data && typeof data === "object" && "refreshToken" in data
            ? (data as { refreshToken?: string }).refreshToken
            : undefined;

        if (accessToken) {
          await setTokens(accessToken, nextRefreshToken ?? refreshToken);
          return accessToken;
        }
        return null;
      } catch {
        return null;
      }
    })().finally(() => {
      refreshPromise = null;
    });
  }

  return refreshPromise;
}

export async function apiFetch(
  path: string,
  options: ApiFetchOptions = {}
): Promise<unknown> {
  const url = `${normalizedBaseUrl}${path}`;
  const isRefreshRequest = path === refreshPath || url.endsWith(refreshPath);

  const buildHeaders = async (overrideToken?: string) => {
    const headers = new Headers(options.headers);

    const isFormData =
      typeof FormData !== "undefined" && options.body instanceof FormData;

    if (!isFormData && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    if (options.auth) {
      const token = overrideToken ?? (await getTokens()).accessToken;
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }
    }

    return headers;
  };

  const requestOnce = async (overrideToken?: string) => {
    const headers = await buildHeaders(overrideToken);

    let response: Response;
    try {
      response = await fetch(url, {
        ...options,
        headers,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Network error fetching ${url}: ${msg}`);
    }

    let data: unknown = null;
    const contentType = response.headers.get("content-type") ?? "";

    try {
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        data = await response.text();
      }
    } catch {
      data = null;
    }

    return { response, data };
  };

  let { response, data } = await requestOnce();

  if (response.status === 401 && options.auth && !isRefreshRequest) {
    const newAccessToken = await refreshAccessToken();

    if (!newAccessToken) {
      await clearTokens();
      if (typeof window !== "undefined") {
        window.location.assign("/signin");
      }
      throw new Error("Session expired. Please sign in again.");
    }

    ({ response, data } = await requestOnce(newAccessToken));
  }

  if (!response.ok) {
    throw new Error(getErrorMessage(data, response));
  }

  return data;
}