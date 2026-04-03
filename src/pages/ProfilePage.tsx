import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ProfileScreen } from "@/components/ProfileScreen";
import { useAppState } from "@/context/AppStateContext";
import type { UserProfile } from "@/types";
import { getProfile, updateProfile } from "@/api/profile";

function resolveProfile(data: unknown): UserProfile | null {
  if (!data || typeof data !== "object") return null;

  const container = data as { profile?: unknown; user?: unknown };
  const candidate =
    (container.profile && typeof container.profile === "object"
      ? container.profile
      : container.user && typeof container.user === "object"
      ? container.user
      : data) ?? null;

  if (!candidate || typeof candidate !== "object") return null;
  const p = candidate as any;

  return {
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
      typeof p.chronicConditions === "string" ? p.chronicConditions : undefined,
    notes: typeof p.notes === "string" ? p.notes : undefined,
  };
}

export function ProfilePage() {
  const navigate = useNavigate();
  const { profile, setProfile, logout } = useAppState();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isActive = true;

    const loadProfile = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await getProfile();
        const resolved = resolveProfile(data);

        if (isActive && resolved) {
          setProfile(resolved);
        }
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err.message : "Failed to load profile.");
        }
      } finally {
        if (isActive) {
          setLoading(false);
        }
      }
    };

    void loadProfile();
    return () => {
      isActive = false;
    };
  }, [setProfile]);

  const handleSave = async (p: UserProfile) => {
    setError(null);
    try {
      setSaving(true);

      // Backend has NOT NULL firstName/lastName, so always send them.
      const dto = {
        firstName: p.firstName ?? "",
        lastName: p.lastName ?? "",
        dob: p.dob,
        gender: p.gender,
        height: p.height,
        weight: p.weight,
        allergies: p.allergies,
        bloodType: p.bloodType,
        chronicConditions: p.chronicConditions,
        notes: p.notes,
      };

      const data = await updateProfile(dto);
      const resolved = resolveProfile(data);
      setProfile(resolved ?? p);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save profile.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center text-sm text-slate-500">
        Loading profile...
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="min-h-[100dvh] flex items-center justify-center text-sm text-slate-500">
        {error ?? "Profile unavailable."}
      </div>
    );
  }

  return (
    <div>
      {error && <div className="px-5 pt-4 text-sm text-red-600">{error}</div>}
      {saving && <div className="px-5 pt-2 text-sm text-slate-500">Saving...</div>}

      <ProfileScreen
        profile={profile}
        onSave={(p) => void handleSave(p)}
        onClose={() => navigate(-1)}
        onLogout={() => {
          void (async () => {
            await logout();
            navigate("/signin", { replace: true });
          })();
        }}
      />
    </div>
  );
}

export default ProfilePage;
