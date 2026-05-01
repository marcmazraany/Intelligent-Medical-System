@echo off
set PATH=%PATH%;C:\ProgramData\chocolatey\bin
where k6
k6 version

@echo off
echo Creating results directory...
mkdir results 2>nul

echo ========================================
echo Running Scenario A: Cache Hit
echo ========================================
k6 run scenario-a-cache-hit.js

echo.
echo ========================================
echo Running Scenario B: Cache Miss
echo ========================================
k6 run scenario-b-cache-miss.js

echo.
echo ========================================
echo Running Scenario C: Mixed Traffic
echo ========================================
k6 run scenario-c-mixed-traffic.js

echo.
echo ========================================
echo All tests completed!
echo Results saved in results/ directory
echo ========================================
pause