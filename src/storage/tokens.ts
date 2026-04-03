import { Preferences } from "@capacitor/preferences";

const ACCESS_TOKEN_KEY = "accessToken";
const REFRESH_TOKEN_KEY = "refreshToken";

type TokenResult = {
  accessToken?: string;
  refreshToken?: string;
};

export async function getTokens(): Promise<TokenResult> {
  const [accessResult, refreshResult] = await Promise.all([
    Preferences.get({ key: ACCESS_TOKEN_KEY }),
    Preferences.get({ key: REFRESH_TOKEN_KEY }),
  ]);

  return {
    accessToken: accessResult.value ?? undefined,
    refreshToken: refreshResult.value ?? undefined,
  };
}

export async function setTokens(
  accessToken: string,
  refreshToken?: string
): Promise<void> {
  const tasks: Promise<void>[] = [
    Preferences.set({ key: ACCESS_TOKEN_KEY, value: accessToken }),
  ];

  if (typeof refreshToken === "string") {
    tasks.push(
      Preferences.set({ key: REFRESH_TOKEN_KEY, value: refreshToken })
    );
  } else {
    tasks.push(Preferences.remove({ key: REFRESH_TOKEN_KEY }));
  }

  await Promise.all(tasks);
}

export async function clearTokens(): Promise<void> {
  await Promise.all([
    Preferences.remove({ key: ACCESS_TOKEN_KEY }),
    Preferences.remove({ key: REFRESH_TOKEN_KEY }),
  ]);
}
