import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, Package, MapPin, Bell, Bot } from 'lucide-react';

const tabs = [
  { to: '/', icon: Home, label: 'Home' },
  { to: '/inventory', icon: Package, label: 'Inventory' },
  { to: '/pharmacies', icon: MapPin, label: 'Find' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/ai', icon: Bot, label: 'AI Help' },
] as const;

export function BottomNav() {
  return (
    <nav className="fixed bottom-0 left-1/2 transform -translate-x-1/2 w-full max-w-[420px] bg-white border border-slate-100 px-2 py-2 pb-safe flex justify-around items-center z-50 shadow-lg rounded-t-3xl">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        return (
          <NavLink
            key={tab.to}
            to={tab.to}
            end={tab.to === '/'}
            className={({ isActive }) =>
              `flex flex-col items-center py-2 px-1 flex-1 transition-colors ${
                isActive ? 'text-blue-600' : 'text-slate-500'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon className={`w-6 h-6 mb-1 ${isActive ? 'text-blue-600' : 'text-slate-400'}`} />
                <span className={`text-[10px] font-medium ${isActive ? 'text-blue-600' : 'text-slate-500'}`}>
                  {tab.label}
                </span>
              </>
            )}
          </NavLink>
        );
      })}
    </nav>
  );
}
