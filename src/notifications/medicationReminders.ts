import { LocalNotifications } from "@capacitor/local-notifications";
import type { Medication } from "@/types";

const SOURCE_TAG = "medication-reminder";
let permissionRequested = false;

export async function requestNotificationPermissionOnce(): Promise<void> {
  if (permissionRequested) return;
  permissionRequested = true;
  try {
    await LocalNotifications.requestPermissions();
  } catch {
    // Ignore permission errors on unsupported platforms.
  }
}

function hashToId(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) || 1;
}

function isActiveMedication(med: Medication): boolean {
  return (med as { active?: boolean }).active ?? true;
}

function parseTime(value: string): { hour: number; minute: number } | null {
  const [hourText, minuteText] = value.split(":");
  const hour = Number(hourText);
  const minute = Number(minuteText);
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return null;
  if (hour < 0 || hour > 23 || minute < 0 || minute > 59) return null;
  return { hour, minute };
}

function buildNotifications(medications: Medication[]) {
  return medications.flatMap((med) => {
    if (!isActiveMedication(med)) return [];
    return med.reminderTimes
      .map((time) => {
        const parsed = parseTime(time);
        if (!parsed) return null;

        return {
          id: hashToId(`${med.id}:${time}`),
          title: "Medication Reminder",
          body: med.dosage ? `${med.name} • ${med.dosage}` : med.name,
          schedule: {
            on: {
              hour: parsed.hour,
              minute: parsed.minute,
            },
            repeats: true,
          },
          extra: {
            source: SOURCE_TAG,
            medicationId: med.id,
            time,
          },
        };
      })
      .filter(
        (notification): notification is NonNullable<typeof notification> =>
          Boolean(notification)
      );
  });
}

export async function syncMedicationNotifications(
  medications: Medication[]
): Promise<void> {
  try {
    const permissions = await LocalNotifications.checkPermissions();
    if (permissions.display !== "granted") {
      return;
    }

    const pending = await LocalNotifications.getPending();
    const toCancel = pending.notifications
      .filter((notification) => notification.extra?.source === SOURCE_TAG)
      .map((notification) => ({ id: notification.id }));

    if (toCancel.length) {
      await LocalNotifications.cancel({ notifications: toCancel });
    }

    const notifications = buildNotifications(medications);
    if (notifications.length) {
      await LocalNotifications.schedule({ notifications });
    }
  } catch {
    // Swallow notification errors to avoid blocking core flows.
  }
}
