import React from 'react';
import { User } from 'lucide-react';

export function Header({
  title,
  subtitle,
  onProfileClick,
}: {
  title: string;
  subtitle?: string;
  onProfileClick?: () => void;
}) {
  return (
    <div className="bg-blue-600 text-white p-6 pt-10 pb-8 rounded-b-[2.5rem] shadow-lg">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          {subtitle && <p className="text-blue-100 text-sm mt-1">{subtitle}</p>}
        </div>
        {onProfileClick && (
          <button
            onClick={onProfileClick}
            className="bg-white/20 p-2 rounded-full backdrop-blur-sm active:scale-90 transition-transform"
            aria-label="Open profile"
          >
            <User className="w-6 h-6 text-white" />
          </button>
        )}
      </div>
    </div>
  );
}
