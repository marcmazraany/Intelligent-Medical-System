import React, { createContext, useContext, useMemo, useEffect } from "react";
import type { Activity, Alert, Medication, UserProfile } from "@/types";
import { useLocalStorageState } from "@/hooks/useLocalStorageState";
import { getTokens, clearTokens } from "@/storage/tokens";
import { getProfile } from "@/api/profile";

type AppState = {
  // profile can be null (logged out)
  profile: UserProfile | null;
  setProfile: React.Dispatch<React.SetStateAction<UserProfile | null>>;

  medications: Medication[];
  setMedications: React.Dispatch<React.SetStateAction<Medication[]>>;

  activities: Activity[];
  alerts: Alert[];

  showProfile: boolean;
  openProfile: () => void;
  closeProfile: () => void;

  logout: () => Promise<void>;
};

const AppStateContext = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [profile, setProfile] = useLocalStorageState<UserProfile | null>(
    "intellimeds.profile",
    null
  );

  const [medications, setMedications] = useLocalStorageState<Medication[]>(
    "intellimeds.meds",
    []
  );

  const [showProfile, setShowProfile] = useLocalStorageState<boolean>(
    "intellimeds.showProfile",
    false
  );

  // Demo-only placeholders (keep empty until you wire backend activity feed)
  const activities: Activity[] = [];
  const alerts: Alert[] = [];

  const openProfile = () => setShowProfile(true);
  const closeProfile = () => setShowProfile(false);

  const logout = async () => {
    await clearTokens();
    try {
      localStorage.removeItem("intellimeds.profile");
    } catch {}
    setProfile(null);
    setMedications([]);
  };

  // Bootstrap session on app start:
  // - If we have an access token, try loading the profile.
  // - If it fails (expired/invalid), clear tokens and stay logged out.
  useEffect(() => {
    let alive = true;

    const bootstrap = async () => {
      const tokens = await getTokens();
      if (!tokens.accessToken) return;

      try {
        const data = await getProfile();
        if (!alive) return;

        // Backend returns a profile object (sometimes nested)
        const candidate =
          (data && typeof data === "object" && "profile" in (data as any)
            ? (data as any).profile
            : data) ?? null;

        if (candidate && typeof candidate === "object") {
          const p = candidate as any;
          const resolved: UserProfile = {
            id: typeof p.id === "string" ? p.id : undefined,
            email: typeof p.email === "string" ? p.email : undefined,
            phone: typeof p.phone === "string" ? p.phone : undefined,
            firstName: typeof p.firstName === "string" ? p.firstName : "",
            lastName: typeof p.lastName === "string" ? p.lastName : "",
            dob: typeof p.dob === "string" ? p.dob : undefined,
            gender: typeof p.gender === "string" ? p.gender : undefined,
            height: typeof p.height === "string" ? p.height : undefined,
            weight: typeof p.weight === "string" ? p.weight : undefined,
            allergies: typeof p.allergies === "string" ? p.allergies : undefined,
            bloodType: typeof p.bloodType === "string" ? p.bloodType : undefined,
            chronicConditions:
              typeof p.chronicConditions === "string"
                ? p.chronicConditions
                : undefined,
            notes: typeof p.notes === "string" ? p.notes : undefined,
          };

          // If first/last name are missing, treat as invalid session
          if (resolved.firstName || resolved.lastName) {
            setProfile(resolved);
          } else {
            await logout();
          }
        } else {
          await logout();
        }
      } catch {
        await logout();
      }
    };

    void bootstrap();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const value = useMemo(
    () => ({
      profile,
      setProfile,
      medications,
      setMedications,
      activities,
      alerts,
      showProfile,
      openProfile,
      closeProfile,
      logout,
    }),
    [profile, medications, showProfile]
  );

  return (
    <AppStateContext.Provider value={value}>
      {children}
    </AppStateContext.Provider>
  );
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used inside AppStateProvider");
  return ctx;
}
