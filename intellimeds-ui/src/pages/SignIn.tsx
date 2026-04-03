import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff } from "lucide-react";
import { useAppState } from "@/context/AppStateContext";
import { signin } from "@/api/auth";
import { getProfile } from "@/api/profile";
import type { UserProfile } from "@/types";

type SignInValues = {
  email: string;
  password: string;
};

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
  const source = candidate as Record<string, unknown>;

  // build name pieces from whatever we have
  let firstName = typeof source.firstName === "string" ? source.firstName : "";
  let lastName = typeof source.lastName === "string" ? source.lastName : "";
  if (!firstName && !lastName && typeof source.name === "string") {
    const parts = source.name.trim().split(/\s+/);
    firstName = parts.shift() || "";
    lastName = parts.join(" ");
  }

  // require at least one of the name pieces; backend always sends at least
  if (!firstName && !lastName) return null;

  return {
    id: typeof source.id === "string" ? source.id : undefined,
    email: typeof source.email === "string" ? source.email : undefined,
    phone: typeof source.phone === "string" ? source.phone : undefined,
    firstName,
    lastName,
    dob: typeof source.dob === "string" ? source.dob : undefined,
    gender: typeof source.gender === "string" ? source.gender : undefined,
    height: typeof source.height === "string" ? source.height : undefined,
    weight: typeof source.weight === "string" ? source.weight : undefined,
    allergies: typeof source.allergies === "string" ? source.allergies : undefined,
    bloodType: typeof source.bloodType === "string" ? source.bloodType : undefined,
    chronicConditions:
      typeof source.chronicConditions === "string" ? source.chronicConditions : undefined,
    notes: typeof source.notes === "string" ? source.notes : undefined,
  };
}

function buildFallbackProfile(email: string, data?: unknown): UserProfile {
  let firstName = email.split("@")[0]?.trim() || "User";
  let lastName = "";

  if (data && typeof data === "object") {
    const container = data as { profile?: unknown; user?: unknown };
    const candidate =
      (container.profile && typeof container.profile === "object"
        ? container.profile
        : container.user && typeof container.user === "object"
        ? container.user
        : data) ?? null;
    if (candidate && typeof candidate === "object") {
      const source = candidate as Record<string, unknown>;
      if (!firstName && typeof source.firstName === "string") {
        firstName = source.firstName;
      }
      if (!lastName && typeof source.lastName === "string") {
        lastName = source.lastName;
      }
      if ((!firstName && !lastName) && typeof source.name === "string") {
        const parts = source.name.trim().split(/\s+/);
        firstName = parts.shift() || "";
        lastName = parts.join(" ");
      }
    }
  }

  return {
    firstName: firstName || "User",
    lastName,
  } as UserProfile; // rest of fields will be undefined
}

export default function SignIn() {
  const navigate = useNavigate();
  const { setProfile } = useAppState();

  const [values, setValues] = useState<SignInValues>({
    email: "",
    password: "",
  });

  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!values.email.trim() || !values.password.trim()) {
      setError("Please enter your email and password.");
      return;
    }

    try {
      setLoading(true);
      const response = await signin({
        email: values.email.trim(),
        password: values.password,
      });

      let profile = resolveProfile(response);
      if (!profile) {
        try {
          const fetched = await getProfile();
          profile = resolveProfile(fetched);
        } catch {}
      }

      setProfile(profile ?? buildFallbackProfile(values.email.trim(), response));

      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign in failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen w-full bg-gray-50 flex items-center justify-center p-5 relative overflow-hidden">
      <BackgroundDecor />

      <div className="relative w-full max-w-[420px]">
        <div className="rounded-[26px] bg-white shadow-xl p-7 sm:p-8">
          <h1 className="text-2xl font-semibold text-gray-900 text-center">
            Sign In
          </h1>
          <p className="text-sm text-gray-500 text-center mt-2">
            Hi! Welcome back, you’ve been missed
          </p>

          <form onSubmit={handleSubmit} className="mt-7 space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700">Email</label>
              <input
                value={values.email}
                onChange={(e) =>
                  setValues((p) => ({ ...p, email: e.target.value }))
                }
                type="email"
                placeholder="example@gmail.com"
                className="mt-2 w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-blue-300"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700">
                Password
              </label>

              <div className="mt-2 relative">
                <input
                  value={values.password}
                  onChange={(e) =>
                    setValues((p) => ({ ...p, password: e.target.value }))
                  }
                  type={showPass ? "text" : "password"}
                  placeholder="••••••••••••"
                  className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3 pr-12 text-sm outline-none focus:ring-2 focus:ring-blue-300"
                />

                <button
                  type="button"
                  onClick={() => setShowPass((s) => !s)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700"
                >
                  {showPass ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>

              <div className="mt-2 flex justify-end">
                <button
                  type="button"
                  className="text-xs text-blue-600 hover:text-blue-700"
                >
                  Forgot Password?
                </button>
              </div>
            </div>

            {error && (
              <div className="rounded-xl bg-red-50 border border-red-100 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-blue-600 py-3 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 disabled:opacity-60"
            >
              {loading ? "Signing in..." : "Sign In"}
            </button>

            <div className="pt-2 text-center text-sm text-gray-600">
              Don’t have an account?{" "}
              <Link
                to="/signup"
                className="text-blue-600 font-semibold hover:underline"
              >
                Sign Up
              </Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

function BackgroundDecor() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">

      <div className="absolute right-[-120px] top-[-120px] h-[320px] w-[320px] rounded-full bg-blue-600" />

      <div className="absolute right-[-160px] top-[-160px] h-[420px] w-[420px] rounded-full bg-gradient-to-br from-blue-400/60 to-blue-300/30" />

    </div>
  );
}

