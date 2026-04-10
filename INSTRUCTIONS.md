# System Design Document: ARMory

> **Agent Instruction:** This document serves as the architectural blueprint and implementation guide for ARMory. Read all sections carefully to understand the system context, required modules, and technical constraints before generating code.

## 1. Overview & Objectives
**System Name:** ARMory (Advanced Robotic Manipulation)
**Primary Objective:** A multithreaded, curses-style CLI tool to concurrently manage, monitor, and command a fleet of robot workstations via SSH and Docker.

**Scope:**
* **In-Scope:** ASCII bootup sequence, interactive CLI dashboard with live status panels, YAML configuration parsing, concurrent SSH execution, localized and centralized logging, and a broadcast command dispatcher with safety confirmations.
* **Out-of-Scope:** Actual cloud server inference backend (dummy implementation only for now), GUI/Web interfaces.

## 2. System Architecture
**High-Level Architecture:** Asynchronous/Concurrent CLI application with cyberpunk vibes. The UI runs on the main thread while a background worker pool handles concurrent SSH polling and command dispatching to avoid blocking the UI.

**Core Technologies & Frameworks:**
* **Language:** Python 3.10+
* **UI Framework:** `curses` (built-in) OR a lightweight modern equivalent (e.g., `Textual` or `rich` for the ASCII art/coloring).
* **Concurrency:** `asyncio` with `asyncssh` (preferred for lean concurrent SSH) OR `concurrent.futures` with `paramiko`.
* **Configuration:** `PyYAML`

## 3. Directory Structure
> **Agent Instruction:** Strictly adhere to the following directory structure when generating files.

```text
armory/
│
├── src/
│   ├── main.py                 # Application entry point & CLI bootup
│   ├── ui/
│   │   └── dashboard.py        # Curses/CLI layout and rendering
│   ├── core/
│   │   ├── config.py           # YAML parser and state management
│   │   └── ssh_client.py       # Concurrent SSH execution logic
│   └── commands/
│       └── dispatcher.py       # Logic for robot commands (enable, disable, etc.)
│
├── logs/                       # Auto-generated directory for logs
│   ├── armory_system.log       # Main system log
│   ├── workstation_1.log       # Robot-specific logs
│   └── workstation_2.log
│
├── config.yaml                 # External configuration file
├── requirements.txt            
└── README.md
```

## 4. Component Details & Implementation Guidelines

### 4.1. Bootup Sequence
1. Clear the terminal.
2. Print "ARMory" in a large, colored ASCII art font.
3. Fetch the current system user (e.g., via `os.getlogin()`) and print: `"Welcome, <whoami>"`.
4. Prompt the user: `"Press any key to proceed..."` before dropping into the main dashboard.

### 4.2. Configuration & State Management (config.yaml)
* Parse `config.yaml` to retrieve a list of robot workstation IDs.
* **Initialization:** Auto-generate a random human-readable name for each robot. Associate it with its consecutive ID (which is the workstation ID from the YAML).
* **State Persistence:** When the dashboard loads, it MUST immediately query the workstations to determine their current state rather than assuming offline.

### 4.3. Robot Status Definitions (Side Panel)
Display a side panel listing robots. Use terminal coloring for status.
* **Red `(offline)`:** Machine is unreachable or Docker container is not running.
* **Yellow `(booted)`:** Machine is reachable via SSH and the Docker container is running.
* **Green `(online)`:** Connected to the server (Triggered by the dummy "Connect to Server" command).

### 4.4. SSH & Docker Execution Logic
> **Agent Instruction:** Ensure `~/.bashrc` is sourced for all remote commands.
* **Connection:** Connect via SSH using the alias/command: `armssh <workstation ID>`.
* **Booting (piper_start):** To start the robot, execute `piper_start`. 
    * *Behavior:* This launches the Docker container and runs `/nethome/rbansal66/CS4803ARM_Lab/user_data/piper_ros/entrypoint.sh`. 
    * *Edge Case:* `piper_start` errors if a container is already running. Catch this gracefully.
* **Attaching (piper_attach):** If the container is already running (detected during state querying), use `piper_attach` for subsequent commands instead of `piper_start`.
* **Optimization:** Maintain a persistent or pooled SSH/Docker-logged-in session state per robot to minimize reconnection latency.

### 4.5. Command Dashboard (Main Panel)
Implement the following broadcast commands. All actions must be executed **concurrently** across all selected/active robots.
* **Enable:** Runs `p enable` -> waits for completion -> runs `p goto init`.
* **Disable:** Runs `p goto reset` -> waits for completion -> runs `p disable`.
* **Goto Init:** Runs `p goto init`.
* **Goto Zero:** Runs `p goto zero`.
* **Connect to Server:** A dummy function/button that updates the robot's UI status to Green `(online)` (simulating client inference connection).

### 4.6. Safety Features
* **Confirmation Prompts:** Any broadcast command (Enable, Disable, Goto Init, Goto Zero) must trigger a `Y/N` confirmation modal or prompt before executing.
* **Clean Exit:** Intercept `Ctrl-C` (`SIGINT`) at all times. Gracefully close SSH connections, save any necessary state, and restore the user's standard terminal view.

## 5. Error Handling & Logging Rules
* **System Log:** Maintain a running log at `armory/logs/armory_system.log` for application-level events (boot, UI errors, config loading).
* **Workstation Logs:** Maintain separate log files for each robot (`armory/logs/<workstation_id>.log`). Route all stdout/stderr from SSH commands to these specific files.
* Do not fail silently if a single robot times out; log the error to its specific file, update its UI status to Red `(offline)`, and continue processing the others.

## 6. Development Constraints
* Keep the codebase as lean as possible. Avoid heavy GUI frameworks or bloated dependencies. 
* Strictly separate UI rendering logic from SSH execution logic to prevent the CLI from freezing during network calls.
