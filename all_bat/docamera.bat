@echo off & setlocal EnableDelayedExpansion
adb shell input tap 523 2090
ping 192.0.2.2 -n 1 -w 60000 > nul
adb shell input tap 523 2090
ping 192.0.2.2 -n 1 -w 2000 > nul