import React, { useState } from 'react';
import { ChevronLeft, Clock, Droplet, Heart, LogOut, Ruler, Scale, ShieldAlert, User } from 'lucide-react';
import type { UserProfile } from '@/types';

/**
 * Full-screen profile overlay (mobile-first).
 * Same UI as your original prototype, refactored into its own component.
 */
export const ProfileScreen: React.FC<{
  profile: UserProfile;
  onSave: (p: UserProfile) => void;
  onClose: () => void;
  onLogout: () => void;
}> = ({ profile, onSave, onClose, onLogout }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState<UserProfile>(profile);

  const handleToggleEdit = () => {
    if (isEditing) {
      onSave(formData);
    }
    setIsEditing(!isEditing);
  };

  const ProfilePane = ({ title, children, colorClass }: { title: string, children: React.ReactNode, colorClass: string }) => (
    <div className={`bg-white rounded-[2rem] p-6 shadow-xl shadow-blue-900/5 border border-slate-100 mb-4 animate-in fade-in slide-in-from-bottom-2 duration-500`}>
      <h3 className={`text-[10px] font-black uppercase tracking-[0.2em] mb-4 flex items-center gap-2 ${colorClass}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-current" />
        {title}
      </h3>
      <div className="space-y-1">
        {children}
      </div>
    </div>
  );

  const EditableField = ({ icon: Icon, label, value, field, colorClass }: { icon: any, label: string, value?: string, field: keyof UserProfile, colorClass: string }) => (
    <div className="flex items-center gap-4 py-3 border-b border-slate-50 last:border-0">
      <div className={`${colorClass} p-2.5 rounded-xl shrink-0`}>
        <Icon className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-0.5">{label}</p>
        {isEditing ? (
          <input 
            className="w-full bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-sm font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
            value={String(formData[field] ?? "")}
            onChange={(e) => setFormData({ ...formData, [field]: e.target.value })}
          />
        ) : (
          <p className="text-sm font-bold text-slate-700 break-words">{value || 'Not specified'}</p>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-[100dvh] bg-slate-50">
      {/* Container for scrollable content including the header */}
      <div className="flex-1 overflow-y-auto no-scrollbar">
          {/* HEADER: Inside the scroll container so it disappears when scrolling */}
          <div className="bg-blue-600 text-white p-6 pt-12 pb-14 rounded-b-[3rem] shadow-xl z-20">
          <div className="flex justify-between items-center mb-6">
            <button onClick={onClose} className="p-2.5 bg-white/10 rounded-full active:scale-90 transition-transform">
              <ChevronLeft className="w-6 h-6" />
            </button>
            <h1 className="text-xl font-bold tracking-tight">Personal Health</h1>
            <button 
              onClick={handleToggleEdit}
              className={`px-6 py-2.5 rounded-full text-[10px] font-black uppercase tracking-widest shadow-lg active:scale-95 transition-all ${
                isEditing ? 'bg-emerald-500 text-white' : 'bg-white text-blue-600'
              }`}
            >
              {isEditing ? 'Save' : 'Edit'}
            </button>
          </div>
          <div className="flex items-center gap-5">
            <div className="w-20 h-20 bg-white/20 rounded-3xl flex items-center justify-center border-2 border-white/30 backdrop-blur-md shadow-inner">
              <User className="w-10 h-10 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-black">{`${profile.firstName} ${profile.lastName}`.trim() || "Your Profile"}</h2>
              <p className="text-blue-100 text-[10px] font-black uppercase tracking-[0.2em] opacity-80 mt-1">Certified Profile</p>
            </div>
          </div>
        </div>

          {/* PANES: Main scrollable body */}
          <div className="px-5 pt-6 pb-12">
          
          <ProfilePane title="Vital Stats" colorClass="text-blue-500">
            <EditableField icon={Clock} label="DOB" value={profile.dob ?? ""} field="dob" colorClass="bg-blue-50 text-blue-500" />
            <EditableField icon={User} label="Gender" value={profile.gender ?? ""} field="gender" colorClass="bg-blue-50 text-blue-500" />
            <EditableField icon={Droplet} label="Blood Type" value={profile.bloodType ?? ""} field="bloodType" colorClass="bg-blue-50 text-blue-500" />
          </ProfilePane>

          <ProfilePane title="Physical Metrics" colorClass="text-emerald-500">
            <EditableField icon={Ruler} label="Height" value={profile.height ?? ""} field="height" colorClass="bg-emerald-50 text-emerald-500" />
            <EditableField icon={Scale} label="Weight" value={profile.weight ?? ""} field="weight" colorClass="bg-emerald-50 text-emerald-500" />
          </ProfilePane>

          <ProfilePane title="Medical Alerts" colorClass="text-orange-500">
            <EditableField icon={ShieldAlert} label="Allergies" value={profile.allergies ?? ""} field="allergies" colorClass="bg-orange-50 text-orange-500" />
            <EditableField icon={Heart} label="Chronic Conditions" value={profile.chronicConditions ?? ""} field="chronicConditions" colorClass="bg-orange-50 text-orange-500" />
          </ProfilePane>

          <ProfilePane title="General Notes" colorClass="text-slate-500">
            <div className="py-2">
              {isEditing ? (
                <textarea 
                  className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-4 text-sm font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                  rows={4}
                  value={formData.notes}
                  onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                />
              ) : (
                <p className="text-sm font-bold text-slate-600 leading-relaxed break-words italic px-1">
                  "{profile.notes || 'No additional notes provided.'}"
                </p>
              )}
            </div>
          </ProfilePane>

          {/* LOGOUT BUTTON: At the bottom of the scroll view */}
          <div className="mt-8 mb-10">
            <button 
              onClick={onLogout}
              className="w-full bg-rose-50 border border-rose-100 text-rose-600 font-black text-xs uppercase tracking-[0.2em] py-5 rounded-[2rem] flex items-center justify-center gap-3 shadow-sm active:scale-[0.98] active:bg-rose-100 transition-all"
            >
              <LogOut className="w-5 h-5" />
              Logout Securely
            </button>
            <p className="text-center text-[9px] text-slate-300 font-bold uppercase tracking-widest mt-6">
              IntelliMeds v2.5.0 • Build 2024
            </p>
          </div>
          </div>
      </div>
    </div>
  );
};

