export type AppRoute = '/' | '/inventory' | '/pharmacies' | '/alerts' | '/ai';

export interface UserProfile {
  id?: string;
  email?: string;
  phone?: string;

  firstName: string;
  lastName: string;

  dob?: string; // ISO date yyyy-mm-dd
  gender?: string;
  height?: string;
  weight?: string;
  allergies?: string;
  bloodType?: string;
  chronicConditions?: string;
  notes?: string;
}

export type MedicationStatus = 'available' | 'low-stock' | 'expiring-soon';

export interface Medication {
  id: string;
  name: string;
  dosage: string;
  expiryDate: string; // yyyy-mm-dd
  frequency: string;
  reminderTimes: string[]; // HH:mm
  status: MedicationStatus;
  notes?: string;
}

export interface Pharmacy {
  id: string;
  name: string;
  location: string;
  distance: string;
  stock: number;
  price: string;
  lastUpdated: string;
  inStock: boolean;
}

export type AlertStatus = 'notified' | 'waiting';

export interface Alert {
  id: string;
  medicationName: string;
  maxPrice: number;
  emailEnabled: boolean;
  createdDate: string;
  lastNotified: string;
  status: AlertStatus;
  active: boolean;
}

export type ActivityType = 'stock' | 'expiry' | 'scan';
export type ActivityIcon = 'check' | 'alert' | 'camera';

export interface Activity {
  id: string;
  type: ActivityType;
  title: string;
  subtitle: string;
  timestamp: string;
  icon: ActivityIcon;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number; // epoch ms
}
