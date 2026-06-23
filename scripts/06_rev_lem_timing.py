import time
from neural_spd.diffusion_streampower_lem import SimpleLem

params = {
    "grid": {
        "source": "create",
        "create_grid": {
            "RasterModelGrid": [
                (300, 100),
                {"xy_spacing": 5},
            ],
        },
    },
    "clock": {"start": 0.0, "stop": 3000000, "step": 500},
    "output": {
        "plot_times": [9999999],
        "save_times": [9999999],
        "report_times": [9999999],
        "save_path": "model_run",
        "fields": None,
        "plot_to_file": False,
    },
    "baselevel": {
        "uplift_rate": 0.001,
    },
    "diffuser": {"D": 0.01},
    "streampower": {"k": 0.001, "m": 0.3, "n": 0.7}
}
start = time.perf_counter()
lem = SimpleLem(params)
while lem.current_time < params["clock"]["stop"]:
    lem.update(params["clock"]["step"])
elapsed = time.perf_counter() - start
print(f"Elapsed time: {elapsed:.2f} seconds")
