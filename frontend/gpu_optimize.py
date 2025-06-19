def get_ollama_gpu_status(selected_model: str, ensure_gpu: bool = False):
    # --- Function to check and activate Ollama gpu model usage ---

    import subprocess
    import requests
    import time

    try:
        r = requests.get("http://localhost:11434/api/ps")
        if not r.ok:
            return "🔴 Could not connect to Ollama API", None, None

        data = r.json()
        models = data.get("models", [])

        if not isinstance(models, list):
            return f"🔴 Unexpected format (not a list): {models}", None, None

        if not models:
            if ensure_gpu:
                subprocess.Popen(["ollama", "run", selected_model, "--keepalive", "30m"])
                return f"✅ {selected_model} is now launching (no models were running)", selected_model, "Launching"
            return "🟡 No active models running", None, None

        # Check if the selected model is already running
        for m in models:
            model_name = m.get("name", "").lower()
            processor = m.get("details", {}).get("quantization_level", "Unknown quantization")
            if selected_model.lower() in model_name:
                return (
                    f'🟢 **{m["name"]}** is already running ({processor})',
                    m["name"],
                    processor
                )

        # If ensure_gpu is True, stop others and launch selected model
        if ensure_gpu:
            for m in models:
                name = m.get("name")
                if name:
                    requests.post("http://localhost:11434/api/stop", json={"name": name})
                    time.sleep(1)

            subprocess.Popen(["ollama", "run", selected_model, "--keepalive", "30m"])
            return f"✅ {selected_model} is now launching (previous models stopped)", selected_model, "Launching"

        return f"🟠 Another model is running (not `{selected_model}`)", None, None

    except Exception as e:
        return f"🔴 Ollama error: {e}", None, None