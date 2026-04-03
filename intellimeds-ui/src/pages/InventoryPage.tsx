import React, { useEffect, useMemo, useState } from 'react';
import { Plus, Search, Pill, Calendar, Bell, Edit3, Trash2, X, Package } from 'lucide-react';
import { Header } from '@/components/Header';
import { useAppState } from '@/context/AppStateContext';
import {
  createMedication,
  deleteMedication,
  listMedications,
  updateMedication,
} from '@/api/medications';
import { syncMedicationNotifications } from '@/notifications/medicationReminders';

type MedicationStatus = 'available' | 'low' | 'expired';

type Medication = {
  id: string;
  name: string;
  dosage: string;
  expiryDate: string;
  frequency: string;
  quantity?: number;
  reminderTimes: string[];
  status?: MedicationStatus;
  notes?: string;
};

type MedicationModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onSave: (medication: Medication) => void;
  initial?: Medication | null;
};

function getMedicationStatus(expiryDate: string): MedicationStatus {
  if (!expiryDate) return 'available';
  const today = new Date();
  const expiry = new Date(expiryDate);

  if (Number.isNaN(expiry.getTime())) return 'available';
  if (expiry < today) return 'expired';

  const diffTime = expiry.getTime() - today.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  if (diffDays <= 30) return 'low';

  return 'available';
}

function normalizeMedication(raw: any): Medication {
  return {
    id: String(raw?.id ?? crypto.randomUUID()),
    name: String(raw?.name ?? ''),
    dosage: String(raw?.dosage ?? ''),
    expiryDate: String(raw?.expiryDate ?? ''),
    frequency: String(raw?.frequency ?? ''),
    quantity:
      typeof raw?.quantity === 'number'
        ? raw.quantity
        : Number(raw?.quantity) > 0
          ? Number(raw.quantity)
          : 1,
    reminderTimes: Array.isArray(raw?.reminderTimes)
      ? raw.reminderTimes.map(String)
      : [],
    status: (raw?.status as MedicationStatus) || getMedicationStatus(String(raw?.expiryDate ?? '')),
    notes: raw?.notes ? String(raw.notes) : undefined,
  };
}

function buildRequest(m: Medication) {
  return {
    name: m.name,
    dosage: m.dosage,
    expiryDate: m.expiryDate,
    frequency: m.frequency,
    quantity: m.quantity ?? 1,
    reminderTimes: m.reminderTimes,
    status: m.status,
    notes: m.notes,
  };
}

function MedicationModal({ isOpen, onClose, onSave, initial }: MedicationModalProps) {
  const [name, setName] = useState(initial?.name ?? '');
  const [dosage, setDosage] = useState(initial?.dosage ?? '');
  const [expiryDate, setExpiryDate] = useState(initial?.expiryDate ?? '');
  const [frequency, setFrequency] = useState(initial?.frequency ?? 'Daily');
  const [quantity, setQuantity] = useState<number>(initial?.quantity ?? 1);
  const [reminderTimes, setReminderTimes] = useState<string[]>(initial?.reminderTimes ?? ['08:00']);
  const [status, setStatus] = useState<MedicationStatus>(initial?.status ?? 'available');
  const [notes, setNotes] = useState(initial?.notes ?? '');

  useEffect(() => {
    if (!isOpen) return;
    setName(initial?.name ?? '');
    setDosage(initial?.dosage ?? '');
    setExpiryDate(initial?.expiryDate ?? '');
    setFrequency(initial?.frequency ?? 'Daily');
    setQuantity(initial?.quantity ?? 1);
    setReminderTimes(initial?.reminderTimes ?? ['08:00']);
    setStatus(initial?.status ?? 'available');
    setNotes(initial?.notes ?? '');
  }, [initial, isOpen]);

  if (!isOpen) return null;

  const handleSave = () => {
    if (!name.trim() || !dosage.trim() || !expiryDate) return;

    onSave({
      id: initial?.id ?? crypto.randomUUID(),
      name: name.trim(),
      dosage: dosage.trim(),
      expiryDate,
      frequency,
      quantity: Math.max(1, quantity || 1),
      reminderTimes: reminderTimes.filter(Boolean),
      status,
      notes: notes.trim() ? notes.trim() : undefined,
    });
  };

  const updateReminderTime = (index: number, value: string) => {
    setReminderTimes((prev) => prev.map((t, i) => (i === index ? value : t)));
  };

  const addReminderTime = () => {
    setReminderTimes((prev) => [...prev, '08:00']);
  };

  const removeReminderTime = (index: number) => {
    setReminderTimes((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-[1px] flex items-end sm:items-center justify-center">
      <div className="w-full sm:max-w-lg bg-white rounded-t-3xl sm:rounded-3xl p-5 shadow-2xl max-h-[92vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-xl font-bold text-slate-800">
              {initial ? 'Edit Medication' : 'Add Medication'}
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Fill in the details below
            </p>
          </div>
          <button
            onClick={onClose}
            className="bg-slate-100 hover:bg-slate-200 rounded-full p-2 transition-colors"
          >
            <X className="w-5 h-5 text-slate-600" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
              Medication Name
            </p>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Panadol"
              className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
                Dosage
              </p>
              <input
                value={dosage}
                onChange={(e) => setDosage(e.target.value)}
                placeholder="e.g. 500mg"
                className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>

            <div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
                Quantity
              </p>
              <input
                type="number"
                min={1}
                step={1}
                value={quantity}
                onChange={(e) => setQuantity(Math.max(1, Number(e.target.value) || 1))}
                placeholder="1"
                className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
                Expiry Date
              </p>
              <input
                type="date"
                value={expiryDate}
                onChange={(e) => setExpiryDate(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>

            <div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
                Frequency
              </p>
              <input
                value={frequency}
                onChange={(e) => setFrequency(e.target.value)}
                placeholder="e.g. Daily"
                className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
              />
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                Reminder Times
              </p>
              <button
                type="button"
                onClick={addReminderTime}
                className="text-xs font-semibold text-blue-600"
              >
                + Add Time
              </button>
            </div>

            <div className="space-y-2">
              {reminderTimes.map((time, index) => (
                <div key={index} className="flex items-center gap-2">
                  <input
                    type="time"
                    value={time}
                    onChange={(e) => updateReminderTime(index, e.target.value)}
                    className="flex-1 bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
                  />
                  {reminderTimes.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeReminderTime(index)}
                      className="bg-slate-100 hover:bg-slate-200 rounded-xl p-3 transition-colors"
                    >
                      <X className="w-4 h-4 text-slate-600" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
              Status
            </p>
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value as MedicationStatus)}
              className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10"
            >
              <option value="available">Available</option>
              <option value="low">Low</option>
              <option value="expired">Expired</option>
            </select>
          </div>

          <div>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">
              Notes
            </p>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              placeholder="Optional notes"
              className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/10 resize-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2">
            <button
              onClick={onClose}
              className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold py-4 rounded-2xl transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 rounded-2xl transition-colors"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function MedicationCard({
  med,
  onEdit,
  onDelete,
}: {
  med: Medication;
  onEdit: (med: Medication) => void;
  onDelete: (med: Medication) => void;
}) {
  const status = med.status || getMedicationStatus(med.expiryDate);

  const statusClasses =
    status === 'expired'
      ? 'bg-red-50 text-red-700 border-red-100'
      : status === 'low'
        ? 'bg-orange-50 text-orange-700 border-orange-100'
        : 'bg-emerald-50 text-emerald-700 border-emerald-100';

  return (
    <div className="bg-white rounded-3xl p-5 shadow-sm border border-slate-100">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0">
          <div className="bg-blue-50 rounded-2xl p-3">
            <Pill className="w-6 h-6 text-blue-600" />
          </div>
          <div className="min-w-0">
            <h3 className="text-base font-bold text-slate-800 truncate">{med.name}</h3>
            <p className="text-sm text-slate-500 mt-1">{med.dosage}</p>
          </div>
        </div>

        <span className={`shrink-0 px-3 py-1 rounded-full text-xs font-semibold border ${statusClasses}`}>
          {status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mt-4">
        <div className="bg-slate-50 rounded-2xl p-3">
          <div className="flex items-center gap-2 text-slate-500">
            <Calendar className="w-4 h-4" />
            <p className="text-[10px] font-bold uppercase tracking-widest">Expiry</p>
          </div>
          <p className="text-sm font-semibold text-slate-800 mt-2">{med.expiryDate}</p>
        </div>

        <div className="bg-slate-50 rounded-2xl p-3">
          <div className="flex items-center gap-2 text-slate-500">
            <Bell className="w-4 h-4" />
            <p className="text-[10px] font-bold uppercase tracking-widest">Frequency</p>
          </div>
          <p className="text-sm font-semibold text-slate-800 mt-2">{med.frequency || '—'}</p>
        </div>

        <div className="bg-slate-50 rounded-2xl p-3">
          <div className="flex items-center gap-2 text-slate-500">
            <Package className="w-4 h-4" />
            <p className="text-[10px] font-bold uppercase tracking-widest">Quantity</p>
          </div>
          <p className="text-sm font-semibold text-slate-800 mt-2">{med.quantity ?? 1}</p>
        </div>

        <div className="bg-slate-50 rounded-2xl p-3">
          <div className="flex items-center gap-2 text-slate-500">
            <Bell className="w-4 h-4" />
            <p className="text-[10px] font-bold uppercase tracking-widest">Reminders</p>
          </div>
          <p className="text-sm font-semibold text-slate-800 mt-2">
            {med.reminderTimes?.length ? med.reminderTimes.join(', ') : '—'}
          </p>
        </div>
      </div>

      {med.notes && (
        <div className="mt-4 bg-slate-50 rounded-2xl p-3">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Notes</p>
          <p className="text-sm text-slate-700 mt-2">{med.notes}</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 mt-4">
        <button
          onClick={() => onEdit(med)}
          className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold py-3 rounded-2xl transition-colors flex items-center justify-center gap-2"
        >
          <Edit3 className="w-4 h-4" />
          Edit
        </button>
        <button
          onClick={() => onDelete(med)}
          className="bg-red-50 hover:bg-red-100 text-red-700 font-semibold py-3 rounded-2xl transition-colors flex items-center justify-center gap-2"
        >
          <Trash2 className="w-4 h-4" />
          Delete
        </button>
      </div>
    </div>
  );
}

export function InventoryPage() {
  const { medications, setMedications } = useAppState();
  const [query, setQuery] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingMedication, setEditingMedication] = useState<Medication | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        setLoading(true);
        const data = await listMedications();
        if (!active) return;

        const next = Array.isArray(data) ? data.map(normalizeMedication) : [];
        setMedications(next as any);
        void syncMedicationNotifications(next as any);
      } catch (err) {
        console.error('Failed to load medications', err);
      } finally {
        if (active) setLoading(false);
      }
    };

    void load();
    return () => {
      active = false;
    };
  }, [setMedications]);

  const filteredMedications = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return medications as Medication[];
    return (medications as Medication[]).filter((med) =>
      [med.name, med.dosage, med.frequency, med.notes]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(q))
    );
  }, [medications, query]);

  const handleAdd = () => {
    setEditingMedication(null);
    setIsModalOpen(true);
  };

  const handleEdit = (med: Medication) => {
    setEditingMedication(med);
    setIsModalOpen(true);
  };

  const handleDelete = async (med: Medication) => {
    try {
      await deleteMedication(med.id);
      setMedications((prev: any) => {
        const next = prev.filter((item: Medication) => item.id !== med.id);
        void syncMedicationNotifications(next);
        return next;
      });
    } catch (err) {
      console.error('Failed to delete medication', err);
    }
  };

  const handleSave = async (medication: Medication) => {
    try {
      if (editingMedication) {
        const saved = await updateMedication(medication.id, buildRequest(medication));
        const normalized = normalizeMedication(saved as any);

        setMedications((prev: any) => {
          const next = prev.map((item: Medication) =>
            item.id === normalized.id ? normalized : item
          );
          void syncMedicationNotifications(next);
          return next;
        });
      } else {
        const saved = await createMedication(buildRequest(medication));
        const normalized = normalizeMedication(saved as any);

        setMedications((prev: any) => {
          const next = [normalized, ...prev];
          void syncMedicationNotifications(next);
          return next;
        });
      }

      setIsModalOpen(false);
      setEditingMedication(null);
    } catch (err) {
      console.error('Failed to save medication', err);
    }
  };

  return (
    <div className="pb-24">
      <Header
        title="Medication Inventory"
        subtitle="Track, update, and manage your medicines"
      />

      <div className="px-5 mt-6">
        <div className="flex gap-3">
          <div className="flex-1 bg-white rounded-2xl px-4 py-3 border border-slate-100 shadow-sm flex items-center gap-3">
            <Search className="w-5 h-5 text-slate-400" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search medications"
              className="w-full bg-transparent outline-none text-sm text-slate-700 placeholder:text-slate-400"
            />
          </div>

          <button
            onClick={handleAdd}
            className="bg-blue-600 hover:bg-blue-700 text-white rounded-2xl px-4 py-3 shadow-sm flex items-center gap-2 font-semibold transition-colors"
          >
            <Plus className="w-5 h-5" />
            Add
          </button>
        </div>

        <div className="mt-6">
          {loading ? (
            <div className="bg-white rounded-3xl p-6 border border-slate-100 shadow-sm text-slate-500 text-center">
              Loading medications...
            </div>
          ) : filteredMedications.length === 0 ? (
            <div className="bg-white rounded-3xl p-8 border border-slate-100 shadow-sm text-center">
              <div className="bg-blue-50 w-14 h-14 rounded-2xl flex items-center justify-center mx-auto">
                <Pill className="w-7 h-7 text-blue-600" />
              </div>
              <h3 className="text-lg font-bold text-slate-800 mt-4">No medications found</h3>
              <p className="text-sm text-slate-500 mt-2">
                Add a medication or try a different search term.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredMedications.map((med) => (
                <MedicationCard
                  key={med.id}
                  med={med}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      <MedicationModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setEditingMedication(null);
        }}
        onSave={handleSave}
        initial={editingMedication}
      />
    </div>
  );
}