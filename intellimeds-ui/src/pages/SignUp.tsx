import React, { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff } from "lucide-react";
import { useAppState } from "@/context/AppStateContext";
import type { UserProfile } from "@/types";
import { signup } from "@/api/auth";
import { getProfile } from "@/api/profile";

type SignUpValues = {
  firstName: string;
  lastName: string;
  phone: string;
  email: string;
  password: string;
  agree: boolean;
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

  let firstName = typeof source.firstName === "string" ? source.firstName : "";
  let lastName = typeof source.lastName === "string" ? source.lastName : "";
  if (!firstName && !lastName && typeof source.name === "string") {
    const parts = source.name.trim().split(/\s+/);
    firstName = parts.shift() || "";
    lastName = parts.join(" ");
  }

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

export default function SignUp() {
  const navigate = useNavigate();
  const { setProfile } = useAppState();

  const [values, setValues] = useState<SignUpValues>({
    firstName: "",
    lastName: "",
    phone: "",
    email: "",
    password: "",
    agree: true,
  });

  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = useMemo(() => {
    return (
      values.firstName.trim() &&
      values.phone.trim() &&
      values.email.trim() &&
      values.password.trim().length >= 6 &&
      values.agree
    );
  }, [values]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!canSubmit) {
      setError("Fill all fields correctly.");
      return;
    }

    try {
      setLoading(true);
      const response = await signup({
        firstName: values.firstName.trim(),
        lastName: values.lastName.trim() || "-",
        email: values.email.trim(),
        phone: values.phone.trim(),
        password: values.password,
      });

      let profile = resolveProfile(response);
      if (!profile) {
        const fetched = await getProfile();
        profile = resolveProfile(fetched);
      }
      if (!profile) {
        throw new Error("Unable to load profile.");
      }
      setProfile(profile);

      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign up failed.");
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
            Create Account
          </h1>

          <form onSubmit={handleSubmit} className="mt-7 space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <input
                value={values.firstName}
                onChange={(e) =>
                  setValues((p) => ({ ...p, firstName: e.target.value }))
                }
                placeholder="First Name"
                className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
              />
              <input
                value={values.lastName}
                onChange={(e) =>
                  setValues((p) => ({ ...p, lastName: e.target.value }))
                }
                placeholder="Last Name"
                className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
              />
            </div>

            <input
              value={values.phone}
              onChange={(e) =>
                setValues((p) => ({ ...p, phone: e.target.value }))
              }
              placeholder="Phone"
              className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
            />

            <input
              value={values.email}
              onChange={(e) =>
                setValues((p) => ({ ...p, email: e.target.value }))
              }
              type="email"
              placeholder="Email"
              className="w-full rounded-xl border border-gray-200 px-4 py-3 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
            />

            <div className="relative">
              <input
                value={values.password}
                onChange={(e) =>
                  setValues((p) => ({ ...p, password: e.target.value }))
                }
                type={showPass ? "text" : "password"}
                placeholder="Password"
                className="w-full rounded-xl border border-gray-200 px-4 py-3 pr-12 text-sm focus:ring-2 focus:ring-blue-300 outline-none"
              />
              <button
                type="button"
                onClick={() => setShowPass((s) => !s)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500"
              >
                {showPass ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>

            {error && (
              <div className="text-sm text-red-600">{error}</div>
            )}

            <button
              type="submit"
              disabled={loading || !canSubmit}
              className="w-full rounded-xl bg-blue-600 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
            >
              {loading ? "Creating..." : "Sign Up"}
            </button>

            <div className="text-center text-sm text-gray-600">
              Already have an account?{" "}
              <Link to="/signin" className="text-blue-600 font-semibold">
                Sign In
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

