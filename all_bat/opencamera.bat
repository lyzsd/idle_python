@echo off & setlocal EnableDelayedExpansion
adb shell input tap 720 2143
ping 192.0.2.2 -n 1 -w 5000 > nul
adb shell input tap 393 1806
ping 192.0.2.2 -n 1 -w 2000 > nul