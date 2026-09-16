"""
MAIA Beacon - Autonomous GPU & NPU Worker Node Entrypoint.
Prints CLI banner and starts uvicorn ASGI server.
"""

from __future__ import annotations

import uvicorn
import config


def print_banner() -> None:
    GREEN = "\033[92m"
    RESET = "\033[0m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

    logo = fr"""
{GREEN}      ###**###      {RESET}
{GREEN}   ##.        .##   {RESET}
{GREEN} ##              ## {RESET}
{GREEN} #.              .# {RESET}   {CYAN}{BOLD} __  __    _    ___    _         ____  _____    _    ____ ___  _   _ {RESET}
{GREEN}#= :####:  :####: =#{RESET}   {CYAN}{BOLD}|  \/  |  / \  |_ _|  / \       | __ )| ____|  / \  / ___/ _ \| \ | |{RESET}
{GREEN}#: ######  ###### :#{RESET}   {CYAN}{BOLD}| |\/| | / _ \  | |  / _ \      |  _ \|  _|   / _ \| |  | | | |  \| |{RESET}
{GREEN}#= :####:  :####: =#{RESET}   {CYAN}{BOLD}| |  | |/ ___ \ | | / ___ \     | |_) | |___ / ___ \ |__| |_| | |\  |{RESET}
{GREEN} #                # {RESET}   {CYAN}{BOLD}|_|  |_/_/   \_\___/_/   \_\    |____/|_____/_/   \_\____\___/|_| \_|{RESET}
{GREEN} ##              ## {RESET}
{GREEN}   ##          ##   {RESET}
{GREEN}     ####++####     {RESET}
"""

    print("===================================================")
    print("  MAIA Beacon - Demarrage du Worker GPU")
    print("===================================================")

    print(logo)
    print(f"[*] Starting MAIA Beacon on {config.BEACON_HOST}:{config.BEACON_PORT}...")
    print(f"[*] Default device target: {config.TARGET_DEVICE}")
    print(f"[*] Models directory: {config.MODELS_DIR}")


def main() -> None:
    print_banner()
    uvicorn.run(
        "api.routes:app",
        host=config.BEACON_HOST,
        port=config.BEACON_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
