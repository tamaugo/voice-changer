#define MyAppName "Lark"
#define MyAppVersion "0.2.0"
#define MyAppExe "voice-changer.exe"

[Setup]
AppId={{8F4C1A2E-6B7D-4E91-9C33-1A2B3C4D5E6F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Lark
DefaultDirName={localappdata}\Lark
DefaultGroupName=Lark
DisableProgramGroupPage=yes
OutputDir=..\target\installer
OutputBaseFilename=Lark-Setup
SetupIconFile=..\assets\lark.ico
UninstallDisplayIcon={app}\lark.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "..\target\release\{#MyAppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs
Source: "..\voices\README.md"; DestDir: "{app}\voices"; Flags: ignoreversion
Source: "..\assets\lark.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\assets\lark.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Lark"; Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\lark.ico"
Name: "{autodesktop}\Lark"; Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\lark.ico"

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "Launch Lark"; Flags: nowait postinstall skipifsilent
