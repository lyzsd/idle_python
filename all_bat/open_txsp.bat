@echo off & setlocal EnableDelayedExpansion
adb shell am start com.tencent.qqlive/com.tencent.qqlive.ona.offline.client.group.DownloadGroupActivity
ping -n 15 127.0.0.1 > nul

