# Neuroblox.jl

| **Documentation** | **Citation** |
|:-------------------------------------------------------------------------------:|:------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] |  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14192732.svg)](https://doi.org/10.5281/zenodo.14192732)

# Installing the Neuroblox GUI

The Neuroblox graphical application is distributed as a downloadable installer for Windows, macOS, and Linux.

## Installation

1. Visit the [Neuroblox.jl releases on GitHub](https://github.com/Neuroblox/Neuroblox.jl/releases).
2. Download the appropriate archive for your operating system:
   - **Windows (x86-64):** `neuroblox-windows-0.9.zip`
   - **macOS (Apple Silicon):** `neuroblox-macos-0.9.zip`
   - **Linux (x86-64):** `neuroblox-linux-0.9.zip`
3. Extract the downloaded ZIP archive.
4. Install the application:
   - **Windows:** Open the extracted folder, then double-click the `.exe` installer and follow the on-screen installation wizard.
   - **macOS:** Open the extracted `.dmg` file, then drag **Neuroblox** into the **Applications** folder.
   - **Linux:** Open the extracted folder, then make the `.AppImage` file executable (`chmod +x`) and double-click it to launch, or run it from a terminal.

## Specific Installation Instructions

### macOS

NeurobloxGUI will launch after the user acknowledges that the software was downloaded from the internet.  The NeurobloxGUI app is code-signed through the Apple Developer Program.

### Windows

Windows may display a **"Windows protected your PC"** SmartScreen warning:

1. On the warning dialog, click **More info**.
2. Click the **Run anyway** button that appears.
3. Follow the installation wizard as normal.

Windows will remember your choice, so the warning will not appear again for this installer.

### Linux

Linux does not block unsigned applications, but the `.AppImage` file must be marked as executable before it can run:

1. In your file manager, right-click the `.AppImage`, open its **Properties**, and enable **Allow executing file as program** (or run `chmod +x neuroblox-*.AppImage` from a terminal).
2. Double-click the file, or launch it from a terminal with `./neuroblox-*.AppImage`.

## System Requirements

### Supported operating systems

- **Windows:** Windows 10 or later (x86-64)
- **macOS:** macOS 12 Monterey or later on Apple Silicon (M1, M2, M3, M4, or newer)
- **Linux:** A modern 64-bit distribution (x86-64) with a desktop environment

### Hardware recommendations

- **Memory:** 16 GB RAM
- **Processor:** 64-bit processor, 4+ CPU cores

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://neuroblox.github.io/NeurobloxDocsHost/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://neuroblox.github.io/NeurobloxDocsHost/stable/

[CI-img]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/Neuroblox/Neuroblox.jl/actions/workflows/CI.yml

