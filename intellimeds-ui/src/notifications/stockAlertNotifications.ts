import { LocalNotifications } from "@capacitor/local-notifications";
import type { Alert } from "@/types";

const SOURCE_TAG = "stock-alert";

/**
 * One-shot notification logic:
 * - When lastNotified first appears (or changes) → fire the notification ONCE
 * - Record it in notifiedCache so it never fires again for the same alert
 * - The alert stays visible on the Alerts page as "Notified" permanently
 */
const notifiedCache = new Map<string, string>();

export async function syncStockAlertNotifications(
  alerts: Alert[]
): Promise<void> {
  try {
    const permissions = await LocalNotifications.checkPermissions();
    if (permissions.display !== "granted") return;

    const toFire: Parameters<typeof LocalNotifications.schedule>[0]["notifications"] = [];

    for (const alert of alerts) {
      if (!alert.lastNotified) continue; // not yet notified by backend — skip

      const alreadyFired = notifiedCache.get(alert.id);

      // Only fire if we haven't fired for this exact lastNotified timestamp yet
      if (alreadyFired === alert.lastNotified) continue;

      // Mark as fired immediately so even if schedule fails we don't double-fire
      notifiedCache.set(alert.id, alert.lastNotified);

      toFire.push({
        id: stableId(alert.id),
        title: `💊 ${alert.medicationName} is in stock!`,
        body: "A nearby pharmacy has it. Tap to find it.",
        schedule: { at: new Date(Date.now() + 300) },
        extra: { source: SOURCE_TAG, alertId: alert.id },
      });
    }

    if (toFire.length > 0) {
      await LocalNotifications.schedule({ notifications: toFire });
    }
  } catch {
    // Swallow — notifications are best-effort
  }
}

function stableId(uuid: string): number {
  let hash = 0;
  for (let i = 0; i < uuid.length; i++) {
    hash = (hash * 31 + uuid.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) || 1;
}
