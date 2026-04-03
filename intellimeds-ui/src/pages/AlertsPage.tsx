import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bell,
  CheckCircle,
  Mail,
  Plus,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { Header } from "@/components/Header";
import { useNavigate } from "react-router-dom";
import type { Alert } from "@/types";
import { createAlert, deleteAlert, listAlerts } from "@/api/alerts";
import { syncStockAlertNotifications } from "@/notifications/stockAlertNotifications";
import { requestNotificationPermissionOnce } from "@/notifications/medicationReminders";

// ── Alert card — no toggle, shows notified state permanently ─────────
function AlertCard({
  alert,
  onDelete,
}: {
  alert: Alert;
  onDelete: (id: string) => void;
}) {
  const isNotified = !!alert.lastNotified;

  return (
    <div className={`bg-white rounded-3xl p-5 shadow-sm border ${isNotified ? "border-emerald-100" : "border-slate-100"}`}>
      {/* Header row */}
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <p className="text-lg font-bold text-slate-800 truncate">
            {alert.medicationName}
          </p>
          <p className="text-slate-500 text-sm mt-1">
            Max price:{" "}
            <span className="font-semibold text-slate-700">
              {Number(alert.maxPrice ?? 0).toLocaleString()} LBP
            </span>
          </p>
          <p className="text-slate-400 text-xs mt-2">
            Created {new Date(alert.createdDate).toLocaleDateString()}
          </p>
        </div>

        {/* Status badge */}
        <div className="shrink-0 ml-3 text-right">
          {isNotified ? (
            <span className="inline-flex items-center gap-1 bg-emerald-50 text-emerald-700 text-xs font-bold px-3 py-1 rounded-full">
              <CheckCircle className="w-3 h-3" />
              Notified
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 bg-slate-50 text-slate-500 text-xs font-bold px-3 py-1 rounded-full">
              <Bell className="w-3 h-3" />
              Watching
            </span>
          )}
          {alert.lastNotified && (
            <p className="text-slate-400 text-xs mt-1">
              {new Date(alert.lastNotified).toLocaleDateString()}
            </p>
          )}
        </div>
      </div>

      {/* Info row */}
      <div className="grid grid-cols-2 gap-3 mt-4">
        <div className="bg-slate-50 rounded-2xl p-3 flex items-center gap-2">
          <Bell className="w-4 h-4 text-slate-400" />
          <div>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              Status
            </p>
            <p className={`text-sm font-semibold mt-1 ${isNotified ? "text-emerald-600" : "text-blue-600"}`}>
              {isNotified ? "Done" : "Active"}
            </p>
          </div>
        </div>
        <div className="bg-slate-50 rounded-2xl p-3 flex items-center gap-2">
          <Mail className="w-4 h-4 text-slate-400" />
          <div>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              Email
            </p>
            <p className="text-sm font-semibold text-slate-700 mt-1">
              {alert.emailEnabled ? "Enabled" : "Disabled"}
            </p>
          </div>
        </div>
      </div>

      {/* Notified message or delete button */}
      <div className="mt-4 flex items-center justify-between gap-3">
        {isNotified ? (
          <div className="flex-1 bg-emerald-50 rounded-2xl px-4 py-3">
            <p className="text-sm text-emerald-700 font-semibold">
              ✓ You were notified when this medication was in stock
            </p>
          </div>
        ) : (
          <div className="flex-1 bg-blue-50 rounded-2xl px-4 py-3">
            <p className="text-sm text-blue-700 font-semibold">
              Watching — we'll notify you when it's available
            </p>
          </div>
        )}

        <button
          onClick={() => onDelete(alert.id)}
          className="bg-red-50 hover:bg-red-100 text-red-500 p-4 rounded-2xl active:opacity-70 transition-colors"
          title="Remove alert"
        >
          <Trash2 className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

// ── Create alert sheet ───────────────────────────────────────────────
function CreateAlertSheet({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (alert: Alert) => void;
}) {
  const [medName, setMedName]           = useState("");
  const [maxPrice, setMaxPrice]         = useState("");
  const [maxDistance, setMaxDistance]   = useState("10");
  const [emailEnabled, setEmailEnabled] = useState(true);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState<string | null>(null);

  const handleCreate = async () => {
    if (!medName.trim()) { setError("Enter a medication name."); return; }
    const price = parseFloat(maxPrice);
    if (isNaN(price) || price <= 0) { setError("Enter a valid max price."); return; }

    try {
      setLoading(true);
      setError(null);
      const created = await createAlert({
        medicationName: medName.trim(),
        maxPrice: price,
        emailEnabled,
        maxDistance: parseFloat(maxDistance) || undefined,
      });
      onCreated(created as Alert);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create alert.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4 py-8">
      <div className="w-full max-w-sm max-h-[calc(100vh-4rem)] overflow-y-auto"> 
        <div className="bg-white rounded-3xl p-6 shadow-2xl">
          <div className="flex items-center justify-between mb-5">
            <p className="font-bold text-slate-800 text-lg">New stock alert</p>
            <button onClick={onClose} className="p-1 rounded-full hover:bg-slate-100">
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                Medication name
              </label>
              <input
                value={medName}
                onChange={(e) => setMedName(e.target.value)}
                placeholder="e.g. Panadol"
                className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-2xl px-4 py-3 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>

            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                Max price (LBP)
              </label>
              <input
                type="number"
                value={maxPrice}
                onChange={(e) => setMaxPrice(e.target.value)}
                placeholder="e.g. 50000"
                className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-2xl px-4 py-3 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>

            <div>
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                Max distance (km)
              </label>
              <input
                type="number"
                value={maxDistance}
                onChange={(e) => setMaxDistance(e.target.value)}
                placeholder="e.g. 10"
                className="mt-1 w-full bg-slate-50 border border-slate-200 rounded-2xl px-4 py-3 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>

            <div className="flex items-center justify-between bg-slate-50 rounded-2xl px-4 py-3">
              <p className="text-sm font-semibold text-slate-700">Email notification</p>
              <button
                type="button"
                onClick={() => setEmailEnabled((v) => !v)}
                className={`w-12 h-6 rounded-full transition-colors relative ${emailEnabled ? "bg-blue-600" : "bg-slate-300"}`}
              >
                <span className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${emailEnabled ? "left-7" : "left-1"}`} />
              </button>
            </div>
          </div>

          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

          <button
            onClick={() => void handleCreate()}
            disabled={loading}
            className="mt-5 w-full bg-blue-600 text-white font-bold py-4 rounded-2xl shadow-lg shadow-blue-500/30 disabled:opacity-60"
          >
            {loading ? "Creating..." : "Create alert"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────
export function AlertsPage() {
  const navigate = useNavigate();
  const [alerts, setAlerts]           = useState<Alert[]>([]);
  const [query, setQuery]             = useState("");
  const [loading, setLoading]         = useState(true);
  const [refreshing, setRefreshing]   = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [showCreate, setShowCreate]   = useState(false);
  const pollRef                       = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadAlerts = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);
    setError(null);

    try {
      const data = await listAlerts();
      const resolved: Alert[] = Array.isArray(data)
        ? (data as Alert[])
        : ((data as { alerts?: Alert[] }).alerts ?? []);

      setAlerts(resolved);

      // Fire local push notifications for newly-notified alerts
      await requestNotificationPermissionOnce();
      await syncStockAlertNotifications(resolved);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load alerts.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { void loadAlerts(); }, [loadAlerts]);

  // Poll every 2 minutes for new notifications
  useEffect(() => {
    pollRef.current = setInterval(() => void loadAlerts(true), 120_000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [loadAlerts]);

  const handleDelete = async (id: string) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
    try {
      await deleteAlert(id);
    } catch {
      void loadAlerts(true);
    }
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return alerts;
    return alerts.filter((a) => a.medicationName.toLowerCase().includes(q));
  }, [alerts, query]);

  // Split into active (watching) and done (notified) for cleaner display
  const watching  = filtered.filter((a) => !a.lastNotified);
  const notified  = filtered.filter((a) => !!a.lastNotified);

  return (
    <div className="pb-24">
      <Header
        title="Alerts"
        subtitle="Get notified when your medication is in stock"
        onProfileClick={() => navigate("/profile")}
      />

      <div className="px-5 mt-6">
        {/* Top bar */}
        <div className="bg-white rounded-3xl p-5 shadow-sm border border-slate-100">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-slate-800 font-bold">Stock alerts</p>
              <p className="text-slate-500 text-sm mt-0.5">
                {watching.length} watching · {notified.length} notified
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => void loadAlerts(true)}
                disabled={refreshing}
                className="bg-slate-100 p-3 rounded-2xl text-slate-500 active:opacity-70"
              >
                <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
              </button>
              <button
                onClick={() => setShowCreate(true)}
                className="bg-blue-600 text-white p-3 rounded-2xl shadow-md shadow-blue-500/30 active:opacity-70"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
          </div>

          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search alerts"
            className="w-full bg-slate-50 border border-slate-200 rounded-2xl px-4 py-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
          />
        </div>

        {loading && (
          <p className="text-sm text-slate-500 mt-4 text-center">Loading alerts...</p>
        )}
        {error && <p className="text-sm text-red-600 mt-4">{error}</p>}

        {/* Watching section */}
        {!loading && watching.length > 0 && (
          <div className="mt-6">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 px-1">
              Watching
            </p>
            <div className="space-y-4">
              {watching.map((a) => (
                <AlertCard key={a.id} alert={a} onDelete={(id) => void handleDelete(id)} />
              ))}
            </div>
          </div>
        )}

        {/* Notified section */}
        {!loading && notified.length > 0 && (
          <div className="mt-6">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 px-1">
              Notified
            </p>
            <div className="space-y-4">
              {notified.map((a) => (
                <AlertCard key={a.id} alert={a} onDelete={(id) => void handleDelete(id)} />
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!loading && filtered.length === 0 && (
          <div className="bg-white rounded-3xl p-8 text-center border border-slate-100 shadow-sm mt-6">
            <CheckCircle className="w-8 h-8 text-emerald-500 mx-auto" />
            <p className="text-slate-800 font-bold mt-3">No alerts yet</p>
            <p className="text-slate-500 text-sm mt-1">
              Tap + to create your first stock alert.
            </p>
          </div>
        )}
      </div>

      {showCreate && (
        <CreateAlertSheet
          onClose={() => setShowCreate(false)}
          onCreated={(newAlert) => {
            setAlerts((prev) => [newAlert, ...prev]);
            setShowCreate(false);
          }}
        />
      )}
    </div>
  );
}
