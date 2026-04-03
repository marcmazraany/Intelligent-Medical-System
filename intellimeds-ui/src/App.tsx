import React, { useEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import { BottomNav } from "@/components/BottomNav";
import { useAppState } from "@/context/AppStateContext";

import { HomePage } from "@/pages/HomePage";
import { InventoryPage } from "@/pages/InventoryPage";
import { PharmaciesPage } from "@/pages/PharmaciesPage";
import { AlertsPage } from "@/pages/AlertsPage";
import { AiHelpPage } from "@/pages/AiHelpPage";
import ProfilePage from "@/pages/ProfilePage";

import SignIn from "@/pages/SignIn";
import SignUp from "@/pages/SignUp";
import { requestNotificationPermissionOnce } from "@/notifications/medicationReminders";

function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const hideNav = location.pathname === "/ai";

  return (
    <div className="min-h-[100dvh] max-w-md mx-auto bg-slate-50 relative flex flex-col overflow-hidden">
      <div className="flex-1 relative overflow-y-auto no-scrollbar">
        {children}
      </div>
      {!hideNav && <BottomNav />}
    </div>
  );
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { profile } = useAppState();
  if (!profile) return <Navigate to="/signin" replace />;
  return <>{children}</>;
}


export default function App() {
  const { profile } = useAppState();
  useEffect(() => {
    void requestNotificationPermissionOnce();
  }, []);

  return (
    <Routes>
      {/* AUTH ROUTES */}
      <Route
        path="/signin"
        element={profile ? <Navigate to="/" replace /> : <SignIn />}
      />
      <Route
        path="/signup"
        element={profile ? <Navigate to="/" replace /> : <SignUp />}
      />

      {/* APP ROUTES (PROTECTED) */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout>
              <HomePage />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/inventory"
        element={
          <ProtectedRoute>
            <Layout>
              <InventoryPage />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/pharmacies"
        element={
          <ProtectedRoute>
            <Layout>
              <PharmaciesPage />
            </Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/alerts"
        element={
          <ProtectedRoute>
            <Layout>
              <AlertsPage />
            </Layout>
          </ProtectedRoute>
        }
      />

      {/* You said you want /ai without BottomNav so keep it standalone */}
      <Route
        path="/ai"
        element={
          <ProtectedRoute>
            <AiHelpPage />
          </ProtectedRoute>
        }
      />

      <Route
        path="/profile"
        element={
          <ProtectedRoute>
            <Layout>
              <ProfilePage />
            </Layout>
          </ProtectedRoute>
        }
      />

      <Route path="*" element={<Navigate to={profile ? "/" : "/signin"} replace />} />

    </Routes>
  );
  
}
