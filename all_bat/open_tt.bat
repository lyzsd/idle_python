@echo off & setlocal EnableDelayedExpansion
adb shell input tap 126 319
ping -n 7 127.0.0.1 > nul
