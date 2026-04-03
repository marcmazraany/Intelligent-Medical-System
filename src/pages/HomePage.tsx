import React, { useRef, useState } from 'react';
import { Camera, MapPin, ChevronRight, CheckCircle, AlertTriangle, Clock } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import type { Activity } from '@/types';
import { Header } from '@/components/Header';
import { useAppState } from '@/context/AppStateContext';
import { scanMedication } from '@/api/medications';
import { syncMedicationNotifications } from '@/notifications/medicationReminders';

function ActivityIcon({ icon }: { icon: Activity['icon'] }) {
  switch (icon) {
    case 'check':
      return <CheckCircle className="w-5 h-5 text-emerald-500" />;
    case 'alert':
      return <AlertTriangle className="w-5 h-5 text-orange-500" />;
    default:
      return <Camera className="w-5 h-5 text-blue-500" />;
  }
}

async function compressImageToJpeg(
  file: File,
  quality = 0.85,
  maxWidth = 1600,
  maxHeight = 1600
): Promise<File> {
  const bitmap = await createImageBitmap(file);

  let width = bitmap.width;
  let height = bitmap.height;

  const scale = Math.min(maxWidth / width, maxHeight / height, 1);
  width = Math.round(width * scale);
  height = Math.round(height * scale);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Could not process image.');
  }

  ctx.drawImage(bitmap, 0, 0, width, height);

  const blob: Blob = await new Promise((resolve, reject) => {
    canvas.toBlob(
      (result) => {
        if (result) resolve(result);
        else reject(new Error('Failed to convert image to JPEG.'));
      },
      'image/jpeg',
      quality
    );
  });

  const safeName = file.name.replace(/\.[^.]+$/, '') || 'scan';
  return new File([blob], `${safeName}.jpg`, { type: 'image/jpeg' });
}

export function HomePage() {
  const navigate = useNavigate();
  const { profile, medications, activities, setMedications } = useAppState();
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const firstName = profile?.firstName?.trim() || 'there';

  const mergeScannedMedications = (data: unknown) => {
    const items =
      data && typeof data === 'object' && Array.isArray((data as { items?: unknown[] }).items)
        ? ((data as { items: Array<{ medication?: unknown }> }).items ?? [])
        : [];

    const scanned = items
      .map((item) => item?.medication)
      .filter(
        (med): med is Record<string, unknown> =>
          !!med && typeof med === 'object' && typeof med.id === 'string'
      );

    if (scanned.length === 0) return;

    setMedications((prev) => {
      const byId = new Map(prev.map((med) => [med.id, med]));
      for (const med of scanned) {
        byId.set(med.id as string, med as any);
      }
      const next = Array.from(byId.values());
      void syncMedicationNotifications(next as any);
      return next as any;
    });
  };

  const handleScanClick = () => {
    setScanError(null);
    fileInputRef.current?.click();
  };

  const handleScanFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';

    if (!file || isScanning) return;

    try {
      setIsScanning(true);
      setScanError(null);

      console.log('original scan file', {
        name: file.name,
        type: file.type,
        size: file.size,
      });

      const normalizedFile = await compressImageToJpeg(file, 0.85, 1600, 1600);

      console.log('normalized scan file', {
        name: normalizedFile.name,
        type: normalizedFile.type,
        size: normalizedFile.size,
      });

      const result = await scanMedication(normalizedFile);
      console.log('scan result', result);

      mergeScannedMedications(result);
      navigate('/inventory');
    } catch (err) {
      console.error('scan failed', err);
      setScanError(err instanceof Error ? err.message : 'Failed to scan barcode.');
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="pb-24">
      <Header
        title={`Welcome back, ${firstName}`}
        subtitle="Manage your medications safely and easily"
        onProfileClick={() => navigate('/profile')}
      />

      <div className="px-5 mt-6">
        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={handleScanClick}
            disabled={isScanning}
            className="bg-blue-500 p-6 rounded-3xl text-white flex flex-col items-center shadow-md active:scale-[0.98] transition-all disabled:opacity-70"
          >
            <div className="bg-white/20 p-3 rounded-full mb-3">
              <Camera className="w-8 h-8" />
            </div>
            <span className="font-semibold text-lg">Scan Meds</span>
            <span className="text-[11px] text-blue-100 mt-1">
              {isScanning ? 'Scanning image…' : 'Use camera or upload image'}
            </span>
          </button>

          <button
            onClick={() => navigate('/pharmacies')}
            className="bg-green-500 p-6 rounded-3xl text-white flex flex-col items-center shadow-md active:scale-[0.98] transition-all"
          >
            <div className="bg-white/20 p-3 rounded-full mb-3">
              <MapPin className="w-8 h-8" />
            </div>
            <span className="font-semibold text-lg">Find Pharmacy</span>
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={handleScanFileChange}
        />

        {scanError && (
          <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {scanError}
          </div>
        )}

        <div className="mt-8 space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-bold text-slate-800">Overview</h2>
            <button
              onClick={() => navigate('/inventory')}
              className="text-blue-600 text-sm font-semibold flex items-center"
            >
              View All <ChevronRight className="w-4 h-4 ml-1" />
            </button>
          </div>

          <div className="bg-white rounded-3xl p-5 shadow-sm border border-slate-100">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-slate-500 text-sm">Total medications</p>
                <p className="text-3xl font-bold text-slate-800 mt-1">{medications.length}</p>
              </div>
              <div className="bg-blue-50 p-4 rounded-2xl">
                <Clock className="w-8 h-8 text-blue-600" />
              </div>
            </div>

            <button
              onClick={() => navigate('/inventory')}
              className="mt-4 w-full bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium py-3 px-4 rounded-2xl flex items-center justify-center transition-colors"
            >
              Manage Inventory
            </button>
          </div>
        </div>

        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-bold text-slate-800">Recent Activity</h2>
          </div>

          <div className="space-y-3">
            {activities.map((activity) => (
              <div
                key={activity.id}
                className="bg-white rounded-2xl p-4 shadow-sm border border-slate-100"
              >
                <div className="flex items-start">
                  <div className="bg-slate-50 p-2 rounded-xl mr-3">
                    <ActivityIcon icon={activity.icon} />
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold text-slate-800">{activity.title}</p>
                    <p className="text-slate-500 text-sm mt-1">{activity.subtitle}</p>
                    <p className="text-slate-400 text-xs mt-2">{activity.timestamp}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}